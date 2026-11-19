"""CPC + Provence fusion as an HTTP service for CooperBench agents.

The agents run in CooperBench's runner process; the scorers need a GPU.
This server loads the same compressor as the LongMemEval/LoCoMo runs
(CPC-Mistral, Provence fused at W=0.25) once, and compresses whatever the
agents send it: an incoming colleague message, or the older part of an
agent's own transcript.

    POST /compress
      {"entries": [{"role": "assistant", "text": "..."}, ...],
       "query": "...", "extra_queries": ["..."],
       "budget": 6000}                        # tokens (cl100k), or
       "ratio": 0.5, "min_tokens": 150}       # keep ratio of the input, never below min_tokens
    -> {"text": ..., "tokens_in": ..., "tokens_out": ..., "units_in": ..., "units_kept": ..., "ms": ...}
    GET /health

Units are built the same way for every input: each entry is split into
lines, each line into sentences, and fragments under four words are merged
into a neighbour (core.units.prepare_units). Lines keep code and tool
output intact as units, which sentence splitting alone would not.

    python -m integrations.cooperbench.compress_server --port 8765
"""

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from core.memory import split_sentences
from core.selection import SelectionConfig
from core.units import Unit, prepare_units
from integrations.cooperbench.render import render_excerpt


def entries_to_units(entries, count_tokens):
    sentences = []
    for index, entry in enumerate(entries):
        role = entry.get("role", "user")
        for line in (entry.get("text") or "").splitlines():
            if not line.strip():
                continue
            for sentence in split_sentences(line):
                sentences.append({"role": role, "text": sentence, "entry_index": index, "meta": {}})
    return prepare_units(sentences, count_tokens)


class Compressor:
    def __init__(self, weight: float, cpc_preset: str, max_seq_length: int):
        from eval.longmemeval import build_tokenwise

        self.tokenwise: Any = build_tokenwise(
            compressor="cpc",
            cpc_preset=cpc_preset,
            cpc_max_seq_length=max_seq_length,
            selection=SelectionConfig(provence_weight=weight),
        )
        self.lock = threading.Lock()  # one GPU, one request at a time

    def compress(self, request: dict) -> dict:
        start = time.perf_counter()
        count = self.tokenwise.count_tokens
        units: list[Unit] = entries_to_units(request.get("entries") or [], count)
        tokens_in = sum(unit.tokens for unit in units)
        if "budget" in request:
            budget = int(request["budget"])
        else:
            budget = max(int(request.get("min_tokens", 0)), int(tokens_in * float(request.get("ratio", 1.0))))

        entries = request.get("entries") or []
        if tokens_in <= budget or not units:
            # Already fits: nothing to drop, no model call, and the text goes back untouched.
            text = "\n\n".join(entry.get("text") or "" for entry in entries) if len(entries) == 1 else None
            text = text if text is not None else render_excerpt(units, entries)
            return self._result(text, tokens_in, units, units, budget, start)
        else:
            with self.lock:
                result = self.tokenwise.compress_units(
                    request.get("query") or "", budget, units, request.get("extra_queries") or ()
                )
            selected = result.selected or []
        return self._result(render_excerpt(selected, entries), tokens_in, units, selected, budget, start)

    def _result(self, text, tokens_in, units, selected, budget, start) -> dict:
        return {
            "text": text,
            "tokens_in": tokens_in,
            "tokens_out": self.tokenwise.count_tokens(text),
            "units_in": len(units),
            "units_kept": len(selected),
            "budget": budget,
            "ms": round(1000 * (time.perf_counter() - start), 1),
        }


def make_handler(compressor: Compressor):
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._reply(200 if self.path == "/health" else 404, {"ok": self.path == "/health"})

        def do_POST(self):
            if self.path != "/compress":
                self._reply(404, {"error": "unknown path"})
                return
            try:
                request = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                self._reply(200, compressor.compress(request))
            except Exception as error:  # report, keep serving
                self._reply(500, {"error": f"{type(error).__name__}: {error}"})

        def log_message(self, format, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--weight", type=float, default=0.25, help="Provence fusion weight")
    parser.add_argument("--cpc-preset", default="mistral")
    parser.add_argument("--cpc-max-seq-length", type=int, default=6144)
    args = parser.parse_args()

    compressor = Compressor(args.weight, args.cpc_preset, args.cpc_max_seq_length)
    # Load both scorers before accepting requests, so the first agent call is not a 60 s model load.
    warmup = "\n".join(f"Line {i} of the warm-up text for the scorers." for i in range(300))
    compressor.compress({"entries": [{"role": "user", "text": warmup}], "budget": 50, "query": "line 7"})
    print(f"compress server ready on {args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), make_handler(compressor)).serve_forever()


if __name__ == "__main__":
    main()
