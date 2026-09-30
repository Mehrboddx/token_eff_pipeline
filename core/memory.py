import pysbd
from google.genai import types

_SEGMENTER = pysbd.Segmenter(language="en", clean=False)


def split_sentences(text):
    """The one sentence splitter used for everything that gets scored.
    Splitting once, here, is what keeps compressor inputs aligned with what
    Memory recorded -- re-splitting a joined string downstream is what used
    to detach role labels and date tags from the sentences they belong to."""
    return [sentence.strip() for sentence in _SEGMENTER.segment(text) if sentence.strip()]


class Memory:
    """Holds the conversation history for one session. Kept separate from
    Agent so the same Agent can drive many independent sessions, each with
    its own Memory instance.

    Every history entry can carry a `meta` dict (e.g. session/time labels
    for compression output, evaluation labels) and a `prefix` that is shown
    to the answering model in the raw history but never scored: metadata is
    attached to sentences as data, not glued into their text."""

    def __init__(self):
        self.history = []
        self.entry_meta = []
        self.sentences = []
        self.turn_boundaries = []  # history indices where a new user turn begins

    def _append_text(self, role, text, meta):
        entry_index = len(self.history) - 1
        self.entry_meta.append(dict(meta or {}))
        if not text:
            return

        self.sentences.extend(
            {"role": role, "text": sentence, "entry_index": entry_index, "meta": self.entry_meta[-1]}
            for sentence in split_sentences(text)
        )

    @staticmethod
    def _parts_to_text(parts):
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

    def add_user_message(self, text, meta=None, prefix=""):
        self.turn_boundaries.append(len(self.history))
        self.history.append(types.Content(role="user", parts=[types.Part.from_text(text=prefix + text)]))
        self._append_text("user", text, meta)

    def add_model_content(self, content, meta=None, scored_text=None):
        """`scored_text` lets a caller that rendered a prefix into `content`
        hand over the unprefixed text for scoring."""
        self.history.append(content)
        text = scored_text if scored_text is not None else self._parts_to_text(getattr(content, "parts", None) or [])
        self._append_text("model", text, meta)

    def add_tool_results(self, parts):
        # The raw Content has role="user" (the only role the API allows for
        # a function-response turn), so the sentence entries carry their own
        # "tool" role to keep it distinguishable from a real user message.
        self.history.append(types.Content(role="user", parts=parts))
        self._append_text("tool", self._parts_to_text(parts), None)

    def get_history(self):
        return self.history

    def get_turn_boundaries(self):
        return list(self.turn_boundaries)

    def get_entry_meta(self, entry_index):
        return self.entry_meta[entry_index]

    def get_sentence_entries(self):
        return list(self.sentences)

    def get_sentences(self, role=None):
        if role is None:
            return [entry["text"] for entry in self.sentences]

        return [entry["text"] for entry in self.sentences if entry["role"] == role]

    def copy(self):
        """Independent copy that shares the (immutable) Content objects --
        lets one expensively-seeded memory be reused across many runs that
        each append their own question."""
        clone = Memory()
        clone.history = list(self.history)
        clone.entry_meta = list(self.entry_meta)
        clone.sentences = list(self.sentences)
        clone.turn_boundaries = list(self.turn_boundaries)
        return clone
