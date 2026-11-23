"""CPC recall: observation masking plus query-conditioned recall of what was masked.

At every step the agent's history is first masked exactly as in "The Complexity
Trap" (SWE-agent's LastNObservations, n = 10): tool outputs older than the last
ten are replaced by "Old environment output: (N lines omitted)". The prefix of
the conversation is therefore the same as the masking baseline's, so prompt
caching behaves the same.

Then CPC-Mem (the compressor service) selects, from all masked outputs, the
lines most relevant to what the agent is doing now: the query is the agent's
latest thought and action. The selection is appended to the newest tool output
as a recall block, each kept excerpt under the command that produced it
(provenance), within `budget` tokens. Different steps recall different parts of
the session; nothing is ever deleted. If the service fails, that step falls back
to plain masking.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from typing import Literal

from pydantic import BaseModel, ConfigDict

_OUTPUT_BLOCK = re.compile(r"^\[step (\d+), output\]$", re.MULTILINE)
RECALL_HEADER = ("[Recalled from earlier outputs that are no longer shown above, selected for your current step. "
                 "Each excerpt follows the command that produced it; '…' marks skipped lines.]")


def _text(entry: dict) -> str:
    content = entry.get("content")
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content or [] if isinstance(part, dict))


def _with_text(entry: dict, text: str) -> dict:
    entry = dict(entry)
    if isinstance(entry.get("content"), list):
        entry["content"] = [dict(entry["content"][0], text=text)] + list(entry["content"][1:])
    else:
        entry["content"] = text
    return entry


def masked_pool(original: list[dict], masked: list[dict]) -> list[dict]:
    """The observations the masking replaced, each with the action that produced it."""
    pool = []
    for index, (before, after) in enumerate(zip(original, masked)):
        if before.get("message_type") == "observation" and _text(before) != _text(after):
            previous = original[index - 1] if index > 0 else {}
            action = (previous.get("action") or "").strip() if previous.get("role") == "assistant" else ""
            pool.append({"action": action, "output": _text(before)})
    return pool


def current_query(history: list[dict]) -> str:
    """What the agent is doing now: its latest thought and action."""
    for entry in reversed(history):
        if entry.get("role") == "assistant":
            parts = [entry.get("thought") or "", entry.get("action") or ""]
            return "\n".join(part.strip() for part in parts if part and part.strip())[:4000]
    return ""


def render_recall(pool: list[dict], excerpt: str, max_action_chars: int = 300) -> str:
    """The service's excerpt ("[step i, output]" blocks, i = pool index + 1) as a recall
    block in which every excerpt sits under its command."""
    parts = _OUTPUT_BLOCK.split(excerpt)
    blocks = []
    for i in range(1, len(parts) - 1, 2):
        index, lines = int(parts[i]) - 1, parts[i + 1].strip()
        if 0 <= index < len(pool) and lines:
            action = pool[index]["action"]
            action = action if len(action) <= max_action_chars else action[:max_action_chars] + " …"
            blocks.append(f"$ {action}\n{lines}" if action else lines)
    return RECALL_HEADER + "\n\n" + "\n\n".join(blocks) if blocks else ""


class CPCRecall(BaseModel):
    type: Literal["cpc_recall"] = "cpc_recall"
    """Do not change. Used for (de)serialization."""

    n: int = 10
    """Masking window: the last n tool outputs stay verbatim (as the masking baseline)."""

    budget: int = 2000
    """Tokens of recalled lines per step (counted by the service, cl100k)."""

    url: str = "http://127.0.0.1:8765"
    timeout: float = 180.0

    model_config = ConfigDict(extra="forbid")

    def __call__(self, history: list[dict]) -> list[dict]:
        from sweagent.agent.history_processors import LastNObservations

        masked = LastNObservations(n=self.n)(history)
        pool = masked_pool(history, masked)
        if not pool or not masked or masked[-1].get("message_type") != "observation":
            return masked
        query = current_query(history)
        record = {"pool": len(pool), "pool_chars": sum(len(item["output"]) for item in pool), "query_chars": len(query)}
        start = time.perf_counter()
        try:
            payload = {"entries": [{"role": "tool", "label": "output", "text": item["output"]} for item in pool],
                       "query": query, "budget": self.budget}
            request = urllib.request.Request(self.url.rstrip("/") + "/compress", data=json.dumps(payload).encode(),
                                             headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read())
            recall = render_recall(pool, result["text"])
            record.update(units_in=result.get("units_in"), units_kept=result.get("units_kept"),
                          tokens_out=result.get("tokens_out"), recall_chars=len(recall))
        except Exception as error:  # the step goes ahead with plain masking
            recall = ""
            record["error"] = f"{type(error).__name__}: {error}"[:300]
        record["ms"] = round(1000 * (time.perf_counter() - start))
        self._log(record)
        if recall:
            masked = masked[:-1] + [_with_text(masked[-1], _text(masked[-1]) + "\n\n" + recall)]
        return masked

    @staticmethod
    def _log(record: dict) -> None:
        path = os.environ.get("CPC_RECALL_LOG")
        if path:
            with open(path, "a", encoding="utf-8") as sink:
                sink.write(json.dumps({"t": round(time.time(), 1), **record}) + "\n")
