from google.genai import types

from core.units import prepare_units


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
        # Validated against LongMemEval (eval/longmemeval.py) -- see that
        # file's --token-budget/--sentence-threshold/--recent-turns defaults.
        compression_sentence_threshold=20,
        compression_token_budget=500,
        recent_turns=4,
        min_unit_words=4,
    ):
        self.agent = agent
        self.memory = memory
        self.tokenwise = tokenwise
        self.max_steps = max_steps
        self.compression_sentence_threshold = compression_sentence_threshold
        self.compression_token_budget = compression_token_budget
        self.recent_turns = recent_turns
        self.min_unit_words = min_unit_words
        # What the most recent compressed history was built from -- the
        # eval reads this to measure which evidence survived compression.
        self.last_compression = None

    def compression_candidates(self):
        """(units, cutoff_index) for the history older than the last
        `recent_turns` turns, or None when compression shouldn't apply.
        Cut on real turn boundaries, not entry count: one turn can span
        several entries (user message, tool round trips, tool results)."""
        if self.tokenwise is None:
            return None

        turn_boundaries = self.memory.get_turn_boundaries()
        if len(turn_boundaries) <= self.recent_turns:
            return None

        cutoff_index = turn_boundaries[-self.recent_turns]
        entries = [entry for entry in self.memory.get_sentence_entries() if entry["entry_index"] < cutoff_index]
        units = prepare_units(entries, self.tokenwise.count_tokens, min_words=self.min_unit_words)
        return units, cutoff_index

    def _build_history(self, query, previous_sentences, retrieval_queries=None):
        self.last_compression = None
        history = self.memory.get_history()

        if self.tokenwise is None or len(previous_sentences) <= self.compression_sentence_threshold:
            return history, "full_history"

        candidates = self.compression_candidates()
        if candidates is None:
            return history, "full_history"
        units, cutoff_index = candidates

        # Scoring query and answering query can differ: the answering model
        # may need framing (e.g. today's date) that would only skew
        # relevance scores if it were scored against too.
        queries = list(retrieval_queries or [query])
        result = self.tokenwise.compress_units(
            query=queries[0],
            token_budget=self.compression_token_budget,
            units=units,
            extra_queries=queries[1:],
        )
        if not result.text:
            return history, "full_history"

        self.last_compression = {"cutoff_index": cutoff_index, "result": result, "candidates": len(units)}
        compressed_content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=f"Relevant earlier context:\n{result.text}")],
        )
        return [compressed_content, *history[cutoff_index:]], "compressed_history"

    def run(self, query, retrieval_queries=None):
        previous_sentences = self.memory.get_sentences()
        self.memory.add_user_message(query)

        model_pass = 0
        for _ in range(self.max_steps):
            model_pass += 1
            history, mode = self._build_history(query, previous_sentences, retrieval_queries)
            response = self.agent.call_model(
                history,
                context_meta={"turn": getattr(self, "current_turn", None), "pass": model_pass, "mode": mode},
            )
            content = response.candidates[0].content
            self.memory.add_model_content(content)

            # content.parts can be None (e.g. an empty/blocked response with
            # no text and no function call) — nothing to act on either way.
            function_calls = [part.function_call for part in (content.parts or []) if part.function_call]
            if not function_calls:
                return response  # agent chose to answer instead of calling a tool: done

            tool_outputs = [self.agent.run_tool(call) for call in function_calls]
            self.memory.add_tool_results(tool_outputs)

        return response  # safety cap: stop after max_steps even if the agent kept going
