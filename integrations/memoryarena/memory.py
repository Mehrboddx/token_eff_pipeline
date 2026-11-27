"""Memory systems for MemoryArena, all behind the benchmark's interface:
`add_chunk(text)` after each session, `wrap_user_prompt(prompt)` before the next.

- long_context, bm25: MemoryArena's own classes, unchanged (all past sessions up to
  120k tokens; the top 3 chunks of at most 2,048 tokens, so at most 6,144 tokens).
- none: no memory; the prompt goes out with the empty-memory block their systems use.
- cpc, random: every past session goes to the compressor service, which keeps the
  lines that fit the budget, ranked by CPC + Provence against the new session's
  prompt, or at random. Memory that already fits comes back whole. The budget
  defaults to bm25's ceiling, 6,144 tokens (o200k, the tokenizer their systems use).
  A JSON entry (travel planning stores one-line JSON) is laid out one field per line
  first, so the units follow its fields; other entries go in unchanged.

The output format is the benchmark's: <memory_context> ... </memory_context>, then
"User: <prompt>".
"""

import json
import urllib.request
import zlib
from functools import lru_cache

CONDITIONS = ("none", "long_context", "bm25", "random", "cpc")
BUDGET = 3 * 2048


@lru_cache(maxsize=1)
def _encoding():
    import tiktoken

    return tiktoken.get_encoding("o200k_base")  # gpt-4o-mini's, as in MemoryArena's memory systems


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text or "", disallowed_special=()))


def wrap(context: str, prompt: str) -> str:
    return "\n".join(["<memory_context>", context or "None", "</memory_context>", f"User: {prompt}"])


def memory_context(wrapped: str) -> str:
    """The text between the memory tags (what the agent was shown as memory)."""
    if "<memory_context>" not in wrapped:
        return ""
    return wrapped.split("<memory_context>", 1)[1].split("</memory_context>", 1)[0].strip()


def readable(chunk: str) -> str:
    """A JSON chunk laid out one field per line, with string values on their own lines,
    so line-based units follow its structure; any other chunk is returned unchanged.
    (Travel memory entries are one-line JSON holding whole tool results.)"""
    try:
        value = json.loads(chunk)
    except (TypeError, ValueError):
        return chunk
    if not isinstance(value, (dict, list)):
        return chunk
    lines: list[str] = []

    def emit(item, indent: int) -> None:
        pad = "  " * indent
        if isinstance(item, dict):
            for key, child in item.items():
                if isinstance(child, (dict, list)) or (isinstance(child, str) and "\n" in child):
                    lines.append(f"{pad}{key}:")
                    emit(child, indent + 1)
                else:
                    lines.append(f"{pad}{key}: {child}")
        elif isinstance(item, list):
            for child in item:
                if isinstance(child, (dict, list)) or (isinstance(child, str) and "\n" in child):
                    lines.append(f"{pad}-")
                    emit(child, indent + 1)
                else:
                    lines.append(f"{pad}- {child}")
        elif isinstance(item, str):
            lines.extend(pad + line for line in item.splitlines())
        else:
            lines.append(f"{pad}{item}")

    emit(value, 0)
    return "\n".join(lines)


class NoMemory:
    def add_chunk(self, chunk: str) -> None:
        pass

    def wrap_user_prompt(self, prompt: str) -> str:
        return wrap("", prompt)


class CompressedMemory:
    """Past sessions, cut to a token budget by the compressor service (CPC or random lines)."""

    def __init__(self, selector: str, url: str, budget: int = BUDGET, timeout: int = 3600):
        self.selector = selector
        self.url = url.rstrip("/")
        self.budget = budget
        self.timeout = timeout
        self.chunks: list[str] = []
        self.last: dict = {}

    def add_chunk(self, chunk: str) -> None:
        self.chunks.append(readable(chunk))

    def wrap_user_prompt(self, prompt: str) -> str:
        if not self.chunks:
            return wrap("", prompt)
        entries = [{"role": "tool", "label": "memory", "step": index + 1, "text": chunk}
                   for index, chunk in enumerate(self.chunks)]
        payload = {"entries": entries, "query": prompt, "dedupe": False}
        if self.selector == "random":
            payload.update(selector="random", seed=zlib.crc32(prompt.encode()))
        budget = self.budget
        for _ in range(4):  # the service counts in cl100k and adds step labels: refit until it fits
            result = self._post({**payload, "budget": budget})
            text = result.get("text") or ""
            over = count_tokens(text) - self.budget
            if over <= 0:
                break
            budget = max(1, budget - over - 8)
        self.last = {"ms": result.get("ms"), "units_in": result.get("units_in"),
                     "units_kept": result.get("units_kept"), "tokens_in": result.get("tokens_in")}
        return wrap(text, prompt)

    def _post(self, payload: dict) -> dict:
        request = urllib.request.Request(self.url + "/compress", data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as reply:
            return json.loads(reply.read())


def make_memory(condition: str, upstream=None, compress_url: str = "", budget: int = BUDGET):
    """A fresh memory for one task (memory starts empty for every task, as in the benchmark).
    `upstream` is integrations.memoryarena.upstream.load() (MemoryArena's own classes)."""
    if condition == "none":
        return NoMemory()
    if condition in ("cpc", "random"):
        return CompressedMemory(condition, compress_url, budget)
    if condition == "long_context":
        return upstream.LongContextMemorySystem()
    if condition == "bm25":
        return upstream.RAGMemorySystem(retrieval_method="bm25")
    raise ValueError(f"unknown condition {condition!r}")
