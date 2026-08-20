import re

from google.genai import types


class Memory:
    """Holds the conversation history for one session. Kept separate from
    Agent so the same Agent can drive many independent sessions, each with
    its own Memory instance."""

    def __init__(self):
        self.history = []
        self.sentences = []

    def _split_sentences(self, text):
        return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text.strip()) if sentence.strip()]

    def _append_text(self, role, text):
        if not text:
            return

        self.sentences.extend(
            {"role": role, "text": sentence}
            for sentence in self._split_sentences(text)
        )

    def _parts_to_text(self, parts):
        text_chunks = []

        for part in parts:
            text = getattr(part, "text", None)
            if text:
                text_chunks.append(text)
                continue

            function_response = getattr(part, "function_response", None)
            if function_response is None:
                continue

            response = getattr(function_response, "response", None)
            if isinstance(response, dict) and "result" in response:
                text_chunks.append(str(response["result"]))
            elif response is not None:
                text_chunks.append(str(response))

        return " ".join(text_chunks)

    def _content_to_text(self, content):
        parts = getattr(content, "parts", None) or []
        return self._parts_to_text(parts)

    def add_user_message(self, text):
        self.history.append(types.Content(role="user", parts=[types.Part.from_text(text=text)]))
        self._append_text("user", text)

    def add_model_content(self, content):
        self.history.append(content)
        self._append_text("model", self._content_to_text(content))

    def add_tool_results(self, parts):
        self.history.append(types.Content(role="user", parts=parts))
        self._append_text("tool", self._parts_to_text(parts))

    def get_history(self):
        return self.history

    def get_sentences(self, role=None):
        if role is None:
            return [entry["text"] for entry in self.sentences]

        return [entry["text"] for entry in self.sentences if entry["role"] == role]