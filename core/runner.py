import re

from google.genai import types


class Runner:
    """Drives the agentic loop: ask the agent for the next step, run any
    tool it requests, feed the result back, and repeat until the agent
    answers directly or the step limit is hit. Neither the agent nor the
    memory knows about looping — that logic lives only here."""

    def __init__(
        self,
        agent,
        memory,
        tokenwise=None,
        max_steps=5,
        compression_sentence_threshold=24,
        compression_token_budget=256,
        recent_turns=4,
    ):
        self.agent = agent
        self.memory = memory
        self.tokenwise = tokenwise
        self.max_steps = max_steps
        self.compression_sentence_threshold = compression_sentence_threshold
        self.compression_token_budget = compression_token_budget
        self.recent_turns = recent_turns

    def _build_history(self, query, previous_sentences):
        history = self.memory.get_history()

        if self.tokenwise is None or len(previous_sentences) <= self.compression_sentence_threshold:
            return history, "full_history"

        if len(history) <= self.recent_turns:
            return history, "full_history"

        history_to_compress = history[:-self.recent_turns]

        token_budget = max(1, self.compression_token_budget // 2)
        user_sentences = self._unique_sentences(self._history_sentences(history_to_compress, {"user"}))
        response_sentences = self._unique_sentences(
            self._history_sentences(history_to_compress, {"model", "tool"}),
            excluded=user_sentences,
        )

        user_context = self.tokenwise.compress(
            query=query,
            token_budget=token_budget,
            sentences=user_sentences,
        )
        response_context = self.tokenwise.compress(
            query=query,
            token_budget=token_budget,
            sentences=response_sentences,
        )

        if not user_context and not response_context:
            return history, "full_history"

        context_parts = []
        if user_context:
            context_parts.append(f"Relevant user context:\n{user_context}")
        if response_context:
            context_parts.append(f"Relevant response context:\n{response_context}")
        compressed_context = types.Content(
            role="user",
            parts=[types.Part.from_text(text="\n\n".join(context_parts))],
        )
        recent_history = history[-self.recent_turns :]
        return [compressed_context, *recent_history], "compressed_history"

    @staticmethod
    def _content_to_text(content):
        parts = getattr(content, "parts", None) or []
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

    @classmethod
    def _history_sentences(cls, history, roles):
        sentences = []
        for content in history:
            role = getattr(content, "role", None)
            if role not in roles:
                continue

            text = cls._content_to_text(content)
            if not text:
                continue

            sentences.extend(sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text.strip()) if sentence.strip())

        return sentences

    @staticmethod
    def _unique_sentences(sentences, excluded=None):
        excluded_set = set(excluded or [])
        unique_sentences = []
        seen = set()

        for sentence in sentences:
            if sentence in excluded_set or sentence in seen:
                continue

            seen.add(sentence)
            unique_sentences.append(sentence)

        return unique_sentences

    def run(self, query):
        previous_sentences = self.memory.get_sentences()
        self.memory.add_user_message(query)

        if self.tokenwise is not None:
            self.tokenwise.set_sentences(self.memory.get_sentences())

        model_pass = 0
        for _ in range(self.max_steps):
            model_pass += 1
            history, mode = self._build_history(query, previous_sentences)
            response = self.agent.call_model(
                history,
                context_meta={"turn": getattr(self, "current_turn", None), "pass": model_pass, "mode": mode},
            )
            content = response.candidates[0].content
            self.memory.add_model_content(content)

            if self.tokenwise is not None:
                self.tokenwise.set_sentences(self.memory.get_sentences())

            function_calls = [part.function_call for part in content.parts if part.function_call]
            if not function_calls:
                return response  # agent chose to answer instead of calling a tool: done

            tool_outputs = [self.agent.run_tool(call) for call in function_calls]
            self.memory.add_tool_results(tool_outputs)

        return response  # safety cap: stop after max_steps even if the agent kept going