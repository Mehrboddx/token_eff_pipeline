from google import genai
from google.genai import types


class Agent:
    """A stateless spec: model, system prompt, and the tools it knows how to
    call. Holds no conversation state — that lives in Memory, not here. This
    means the same Agent can be reused across many independent conversations."""

    def __init__(self, name, model, system_prompt, project, location="us-central1", tools=None):
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools or {}  # {name: Tool}
        self.client = genai.Client(vertexai=True, project=project, location=location)

    def _config(self):
        declarations = [tool.declaration for tool in self.tools.values()]
        tool_list = [types.Tool(function_declarations=declarations)] if declarations else None
        return types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            tools=tool_list,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    def call_model(self, history):
        """Given a conversation history, ask the model for the next step."""
        return self.client.models.generate_content(
            model=self.model,
            contents=history,
            config=self._config(),
        )

    def run_tool(self, function_call):
        """Execute a tool the model asked for and package up the result."""
        tool = self.tools[function_call.name]
        result = tool.run(**function_call.args)
        return types.Part.from_function_response(
            name=function_call.name,
            response={"result": result},
        )