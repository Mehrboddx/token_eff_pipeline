"""LLM spend so far, from every agent ledger under a log root (finished or
still running). Prints one number, in USD, for the queue's spend cap.

    python3 -m integrations.cooperbench.spend logs_cooperbench
"""

import json
import sys
from pathlib import Path

from integrations.cooperbench.report import PRICE_CACHED, PRICE_INPUT, PRICE_OUTPUT


def spend(root: Path) -> float:
    total = 0.0
    for ledger in root.glob("**/*_ledger.jsonl"):
        for line in ledger.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:  # a line being written right now
                continue
            if row.get("event") != "llm":
                continue
            cached = row.get("cached_tokens") or 0
            uncached = (row.get("prompt_tokens") or 0) - cached
            total += uncached * PRICE_INPUT + cached * PRICE_CACHED + (row.get("completion_tokens") or 0) * PRICE_OUTPUT
    return total / 1e6


if __name__ == "__main__":
    print(f"{spend(Path(sys.argv[1] if len(sys.argv) > 1 else 'logs_cooperbench')):.2f}")
