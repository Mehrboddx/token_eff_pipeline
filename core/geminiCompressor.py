import re
from typing import Optional

from google import genai
from google.genai import types


class GeminiCompressor:
    """Uses a Gemini model itself (instead of a local CPC/extractive model)
    to compress context down to what's relevant to a query. Matches the
    `model.compress(query, token_budget, context)` interface TokenWise
    delegates to, so it can be dropped in as `TokenWise(model=GeminiCompressor(...))`."""

    PROMPT_TEMPLATE = (
        "You compress conversation context for another AI agent. Given the "
        "query below and a block of prior context, extract and condense only "
        "the sentences relevant to answering the query. Preserve original "
        "wording where possible instead of paraphrasing. Target roughly {token_budget} "
        "tokens of output — do not pad to reach it, shorter is fine if that's "
        "all that's relevant. Output only the compressed context, no preamble "
        "or explanation.\n\n"
        "Query: {query}\n\n"
        "Context:\n{context}"
    )

    def __init__(
        self,
        project: Optional[str] = None,
        location: str = "us-central1",
        model: str = "gemini-2.5-flash",
        client: Optional[genai.Client] = None,
    ) -> None:
        self.model = model
        self.client = client or genai.Client(vertexai=True, project=project, location=location)
        # Real token counting for _enforce_budget instead of word count --
        # a "2000" budget was previously ~2000 words, which for English
        # text is closer to ~2600 real tokens (words * ~1.3), meaning this
        # compressor was silently getting a noticeably larger effective
        # context than a CPC compressor given the same nominal token_budget.
        import tiktoken

        self._tokenizer = tiktoken.encoding_for_model("gpt-4")

    def _config(self) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(temperature=0)

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]

    def _count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text))

    def _truncate_to_budget(self, sentence: str, token_budget: int) -> str:
        if token_budget <= 0:
            return ""
        token_ids = self._tokenizer.encode(sentence)[:token_budget]
        return self._tokenizer.decode(token_ids).strip() + "…"

    def _enforce_budget(self, text: str, token_budget: int) -> str:
        """Gemini is asked to target token_budget but isn't guaranteed to
        respect it exactly, so trim with the same greedy approach
        TokenWise's local fallback uses, keeping sentence order intact.
        Two fixes over the original version: counts real tokens (tiktoken)
        rather than words, and truncates (rather than including in full)
        a first sentence that alone exceeds the budget -- the same
        overrun bug already fixed in TokenWise's local compressor."""
        sentences = self._split_sentences(text)
        if not sentences:
            return ""

        selected: list[str] = []
        used = 0
        for sentence in sentences:
            tokens = self._count_tokens(sentence)

            if not selected and tokens > token_budget:
                sentence = self._truncate_to_budget(sentence, token_budget)
                tokens = self._count_tokens(sentence)

            if selected and used + tokens > token_budget:
                break

            selected.append(sentence)
            used += tokens
            if used >= token_budget:
                break

        return " ".join(selected)

    def compress(self, query: str, token_budget: int, context: str) -> str:
        if not context or not context.strip():
            return ""

        prompt = self.PROMPT_TEMPLATE.format(token_budget=token_budget, query=query, context=context)
        response = self.client.models.generate_content(
            model=self.model,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
            config=self._config(),
        )

        return self._enforce_budget(response.text or "", token_budget)
