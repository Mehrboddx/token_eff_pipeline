"""LLM calls (LiteLLM) with usage recorded, and o200k token counting as in the paper."""

import time
from functools import lru_cache


@lru_cache(maxsize=1)
def _encoding():
    import tiktoken

    return tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text or "", disallowed_special=()))


def complete(model: str, messages: list[dict], max_tokens: int, reasoning_effort: str | None = None,
             temperature: float | None = 0.0, retries: int = 4) -> dict:
    """One chat call -> {"text", "prompt_tokens", "completion_tokens", "cached_tokens", "error"}."""
    import litellm

    kwargs = {"model": model, "messages": messages, "max_tokens": max_tokens, "top_p": 1.0}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    error = ""
    for attempt in range(retries):
        try:
            response = litellm.completion(**kwargs)
            usage = response.get("usage") or {}
            details = usage.get("prompt_tokens_details") or {}
            return {
                "text": response.choices[0].message.content or "",
                "prompt_tokens": usage.get("prompt_tokens") or 0,
                "completion_tokens": usage.get("completion_tokens") or 0,
                "cached_tokens": (details.get("cached_tokens") if isinstance(details, dict) else getattr(details, "cached_tokens", 0)) or 0,
                "error": "",
            }
        except Exception as caught:  # rate limits and transient API errors: back off and retry
            error = f"{type(caught).__name__}: {caught}"[:500]
            time.sleep(min(60, 5 * 2 ** attempt))
    return {"text": "", "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "error": error}
