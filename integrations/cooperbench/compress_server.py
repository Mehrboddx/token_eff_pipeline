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

Ablations (same units, budget packing and rendering; only the ranking changes):
       "selection": {"provence_weight": 1.0, "lexical_weight": 0.0}   # this request only
       "selector": "random", "seed": 0                                  # random ranking
       "dedupe": false          # keep exact repeats (default: only the first occurrence)
       "headings": true         # each kept unit brings its entry's first unit and the first unit
                                # of its indented block (a table's header row, a plan's title)
    An entry may also carry "step" (its number in the excerpt labels) and "label".

Units are built the same way for every input (entries_to_units): lines,
split into sentences where a line holds several; pieces under four words
join the unit before them, by a newline, so code and tool output keep
their layout. tokens_in is the raw text the agent would have read.

    python -m integrations.cooperbench.compress_server --port 8765
"""

import argparse
import dataclasses
import json
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from core.memory import split_sentences
from core.selection import SelectionConfig, select_units

SELECTION_OVERRIDES = {"provence_weight", "lexical_weight"}
from core.units import Unit, word_count
from integrations.cooperbench.render import render_excerpt


def entries_to_units(entries, count_tokens, min_words: int = 4, max_words: int = 60, dedupe: bool = True) -> list[Unit]:
    return units_with_indents(entries, count_tokens, min_words, max_words, dedupe)[0]


def unit_headings(units: list[Unit], indents: list[int]) -> dict[int, set[int]]:
    """Unit id -> the units that head it: the first unit of its entry, and the first unit
    of its indented block (the run of lines at its indentation under a less-indented
    line), e.g. a table's header row, or a plan's title line."""
    heads: dict[int, set[int]] = {}
    stack: list[tuple[int, int]] = []  # (indentation, id of the block's first unit)
    entry, first = None, None
    for unit, indent in zip(units, indents):
        if unit.entry_index != entry:
            entry, first, stack = unit.entry_index, unit.id, []
        while stack and stack[-1][0] > indent:
            stack.pop()
        if not stack or stack[-1][0] < indent:
            stack.append((indent, unit.id))
        heads[unit.id] = {first, stack[-1][1]} - {unit.id}
    return heads


def units_with_indents(entries, count_tokens, min_words: int = 4, max_words: int = 60,
                       dedupe: bool = True) -> tuple[list[Unit], list[int]]:
    """Entries -> scorable units, the same way for every input.

    Each line is a piece (split further into sentences when it holds
    several), with its indentation kept. A piece under `min_words` words
    (`}`, `import os`, "Thanks!") joins the unit before it, by a newline
    when it is its own line, so code keeps its layout and short lines are
    not nearly-free picks under a budget; a unit stops absorbing pieces at
    `max_words`. Exact repeats keep only their first occurrence, unless
    `dedupe` is off (the same line at different steps can mean different things)."""
    units: list[Unit] = []
    indents: list[int] = []  # indentation of the line each unit starts on
    seen: set[tuple[str, str]] = set()
    for index, entry in enumerate(entries):
        role = entry.get("role", "user")
        pieces: list[tuple[str, bool, int]] = []  # (text, starts a line, the line's indentation)
        for line in (entry.get("text") or "").splitlines():
            if not line.strip():
                continue
            sentences = split_sentences(line)
            if len(sentences) <= 1:
                sentences = [line.rstrip()]
            indent = len(line) - len(line.lstrip(" "))
            pieces.extend((sentence, k == 0, indent) for k, sentence in enumerate(sentences))
        merged: list[list] = []
        for text, new_line, indent in pieces:
            if merged and word_count(text) < min_words and word_count(merged[-1][0]) < max_words:
                merged[-1][0] += ("\n" if new_line else " ") + text
            else:
                merged.append([text, indent])
        for text, indent in merged:
            key = (role, text.strip())
            if dedupe and key in seen:
                continue
            seen.add(key)
            units.append(Unit(id=len(units), text=text, role=role, entry_index=index, tokens=count_tokens(text)))
            indents.append(indent)
    return units, indents


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
        entries = request.get("entries") or []
        units, indents = units_with_indents(entries, count, dedupe=bool(request.get("dedupe", True)))
        tokens_in = sum(count(entry.get("text") or "") for entry in entries)  # what the agent would have read
        if "budget" in request:
            budget = int(request["budget"])
        else:
            budget = max(int(request.get("min_tokens", 0)), int(tokens_in * float(request.get("ratio", 1.0))))

        if tokens_in <= budget or not units:
            # Already fits: nothing to drop, no model call, and the text goes back untouched.
            text = entries[0].get("text") or "" if len(entries) == 1 else render_excerpt(units, entries)
            return self._result(text, tokens_in, units, units, budget, start)
        if not request.get("headings"):
            selected = self._choose(request, units, budget)
            return self._result(render_excerpt(selected, entries), tokens_in, units, selected, budget, start)
        # Headings: every kept unit brings the units that head it; select again with less
        # budget until the whole excerpt fits.
        heads = unit_headings(units, indents)
        inner = budget
        for _ in range(6):
            chosen = self._choose(request, units, inner)
            ids = {unit.id for unit in chosen} | {head for unit in chosen for head in heads[unit.id]}
            selected = [units[index] for index in sorted(ids)]
            over = count(render_excerpt(selected, entries)) - budget
            if over <= 0:
                break
            inner = max(1, inner - over)
        else:  # no room for headings at this budget: plain selection
            selected = self._choose(request, units, budget)
        return self._result(render_excerpt(selected, entries), tokens_in, units, selected, budget, start)

    def _choose(self, request: dict, units: list[Unit], budget: int) -> list[Unit]:
        if request.get("selector") == "random":  # ablation floor: no scorer, random ranking
            rng = random.Random(int(request.get("seed", 0)))
            ids = select_units(units, [rng.random() for _ in units], budget, self.tokenwise.count_tokens,
                               self.tokenwise.selection)
            return [units[index] for index in ids]
        overrides = {key: float(value) for key, value in (request.get("selection") or {}).items()
                     if key in SELECTION_OVERRIDES}
        with self.lock:
            saved = self.tokenwise.selection
            self.tokenwise.selection = dataclasses.replace(saved, **overrides)
            try:
                result = self.tokenwise.compress_units(
                    request.get("query") or "", budget, units, request.get("extra_queries") or ()
                )
            finally:
                self.tokenwise.selection = saved
        return list(result.selected or [])

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
