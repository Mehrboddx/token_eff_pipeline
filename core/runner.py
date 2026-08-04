class Runner:
    """Drives the agentic loop: ask the agent for the next step, run any
    tool it requests, feed the result back, and repeat until the agent
    answers directly or the step limit is hit. Neither the agent nor the
    memory knows about looping — that logic lives only here."""

    def __init__(self, agent, memory, max_steps=5):
        self.agent = agent
        self.memory = memory
        self.max_steps = max_steps

    def run(self, query):
        self.memory.add_user_message(query)

        for _ in range(self.max_steps):
            response = self.agent.call_model(self.memory.get_history())
            content = response.candidates[0].content
            self.memory.add_model_content(content)

            function_calls = [part.function_call for part in content.parts if part.function_call]
            if not function_calls:
                return response  # agent chose to answer instead of calling a tool: done

            tool_outputs = [self.agent.run_tool(call) for call in function_calls]
            self.memory.add_tool_results(tool_outputs)

        return response  # safety cap: stop after max_steps even if the agent kept going