from google.genai import types


class Memory:
    """Holds the conversation history for one session. Kept separate from
    Agent so the same Agent can drive many independent sessions, each with
    its own Memory instance."""

    def __init__(self):
        self.history = []

    def add_user_message(self, text):
        self.history.append(types.Content(role="user", parts=[types.Part.from_text(text=text)]))

    def add_model_content(self, content):
        self.history.append(content)

    def add_tool_results(self, parts):
        self.history.append(types.Content(role="user", parts=parts))

    def get_history(self):
        return self.history