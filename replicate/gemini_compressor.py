"""
Gemini-based compressor: same public interface as CPCCompressor
(`compress(context, question, compression_target_tokens) -> str`), so it can
be swapped in via run_long_bench.py's --compressor flag. Instead of running
a local embedding model forward pass per chunk, this delegates sentence
selection to a hosted Gemini model over Vertex AI.
"""

import re

from google import genai
from google.genai import types


class GeminiCompressor:
    PROMPT_TEMPLATE = (
        "You compress documents for a downstream QA model. Given the "
        "question below and a block of context, extract and condense only "
        "the sentences relevant to answering the question. Preserve original "
        "wording where possible instead of paraphrasing. Target roughly "
        "{compression_target_tokens} words of output — do not pad to reach "
        "it, shorter is fine if that's all that's relevant. Output only the "
        "compressed context, no preamble or explanation.\n\n"
        "Question: {question}\n\n"
        "Context:\n{context}"
    )

    def __init__(
        self,
        project: str | None = None,
        location: str = "us-central1",
        model: str = "gemini-2.5-flash",
        client: genai.Client | None = None,
    ) -> None:
        self.model = model
        self.client = client or genai.Client(vertexai=True, project=project, location=location)

    def _config(self) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(temperature=0)

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]

    def _enforce_budget(self, text: str, compression_target_tokens: int) -> str:
        """Gemini is asked to target compression_target_tokens words but isn't
        guaranteed to respect it exactly, so trim greedily by word count,
        keeping sentence order intact."""
        sentences = self._split_sentences(text)
        if not sentences:
            return ""

        selected: list[str] = []
        used = 0
        for sentence in sentences:
            words = len(sentence.split())
            if selected and used + words > compression_target_tokens:
                break
            selected.append(sentence)
            used += words
            if used >= compression_target_tokens:
                break

        return " ".join(selected)

    def compress(self, context: str, question: str, compression_target_tokens: int) -> str:
        if not context or not context.strip():
            return context

        prompt = self.PROMPT_TEMPLATE.format(
            compression_target_tokens=compression_target_tokens,
            question=question,
            context=context,
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
            config=self._config(),
        )

        return self._enforce_budget(response.text or "", compression_target_tokens)
