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
        "words of output — do not pad to reach it, shorter is fine if that's "
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

    def _config(self) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(temperature=0)

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]

    def _enforce_budget(self, text: str, token_budget: int) -> str:
        """Gemini is asked to target token_budget but isn't guaranteed to
        respect it exactly, so trim with the same greedy word-count approach
        TokenWise's local fallback uses, keeping sentence order intact."""
        sentences = self._split_sentences(text)
        if not sentences:
            return ""

        selected: list[str] = []
        used = 0
        for sentence in sentences:
            words = len(sentence.split())
            if selected and used + words > token_budget:
                break
            selected.append(sentence)
            used += words
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
