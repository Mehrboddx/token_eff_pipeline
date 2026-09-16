from google.genai import types


class Runner:
    """Drives the agentic loop: ask the agent for the next step, run any
    tool it requests, feed the result back, and repeat until the agent
    answers directly or the step limit is hit. Neither the agent nor the
    memory knows about looping — that logic lives only here."""

    ROLE_LABELS = {"user": "User", "model": "Assistant", "tool": "Tool"}

    def __init__(
        self,
        agent,
        memory,
        tokenwise=None,
        max_steps=5,
        # Validated against LongMemEval (eval/longmemeval.py): haystacks run
        # tens of thousands of words, and these are the values that
        # actually preserved answer-bearing content through compression —
        # see that file's --token-budget/--sentence-threshold/--recent-turns
        # defaults, which mirror these.
        compression_sentence_threshold=20,
        compression_token_budget=500,
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

        turn_boundaries = self.memory.get_turn_boundaries()
        if len(turn_boundaries) <= self.recent_turns:
            return history, "full_history"

        # Slice on real turn boundaries, not raw history-entry count: a
        # single turn can span several entries (user message, one model
        # call per tool round trip, each tool result), so counting entries
        # instead of turns could cut a "recent" turn off mid-way.
        cutoff_index = turn_boundaries[-self.recent_turns]
        recent_history = history[cutoff_index:]

        # Pull sentences straight from Memory, which already tags each one
        # with its real role (user/model/tool) and keeps them in
        # chronological order — unlike the raw Content list, where a tool
        # result is stored with role="user" (the API's only option for a
        # function-response turn) and would otherwise be mistaken for
        # something the user said.
        labeled_sentences = self._unique_sentences(
            f"{self.ROLE_LABELS.get(entry['role'], entry['role'])}: {entry['text']}"
            for entry in self.memory.get_sentence_entries()
            if entry["entry_index"] < cutoff_index
        )

        compressed_context = self.tokenwise.compress(
            query=query,
            token_budget=self.compression_token_budget,
            sentences=labeled_sentences,
        )

        if not compressed_context:
            return history, "full_history"

        compressed_content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=f"Relevant earlier context:\n{compressed_context}")],
        )
        return [compressed_content, *recent_history], "compressed_history"

    @staticmethod
    def _unique_sentences(sentences):
        unique_sentences = []
        seen = set()

        for sentence in sentences:
            if sentence in seen:
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

            # content.parts can be None (e.g. an empty/blocked response with
            # no text and no function call) — nothing to act on either way.
            function_calls = [part.function_call for part in (content.parts or []) if part.function_call]
            if not function_calls:
                return response  # agent chose to answer instead of calling a tool: done

            tool_outputs = [self.agent.run_tool(call) for call in function_calls]
            self.memory.add_tool_results(tool_outputs)

        return response  # safety cap: stop after max_steps even if the agent kept going