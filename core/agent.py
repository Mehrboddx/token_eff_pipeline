from google import genai
from google.genai import types


class Agent:
    """A stateless spec: model, system prompt, and the tools it knows how to
    call. Holds no conversation state — that lives in Memory, not here. This
    means the same Agent can be reused across many independent conversations."""

    def __init__(
        self,
        name,
        model,
        system_prompt,
        project,
        location="us-central1",
        tools=None,
        context_monitor=None,
    ):
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools or {}  # {name: Tool}
        self.context_monitor = context_monitor
        self.client = genai.Client(vertexai=True, project=project, location=location)

    @staticmethod
    def _content_text(content):
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

    @staticmethod
    def _content_kind(content):
        parts = getattr(content, "parts", None) or []
        if not parts:
            return getattr(content, "role", None)

        if any(getattr(part, "function_response", None) is not None for part in parts):
            return "tool"

        text = Agent._content_text(content)
        if text.startswith("Relevant user context:\n") or text.startswith("Relevant response context:\n"):
            return "compressed_context"

        return getattr(content, "role", None)

    @classmethod
    def describe_history(cls, history):
        return [
            {
                "role": getattr(content, "role", None),
                "kind": cls._content_kind(content),
                "text": cls._content_text(content),
            }
            for content in history
        ]

    def _config(self):
        declarations = [tool.declaration for tool in self.tools.values()]
        tool_list = [types.Tool(function_declarations=declarations)] if declarations else None
        return types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            tools=tool_list,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    def call_model(self, history, context_meta=None):
        """Given a conversation history, ask the model for the next step."""
        if self.context_monitor is not None:
            self.context_monitor(self.describe_history(history), context_meta=context_meta)

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