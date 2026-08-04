from google import genai
from google.genai import types


class Agent:
    def __init__(self, name, model, system_prompt, project, location="global", tools=None):
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.client = genai.Client(vertexai=True, project=project, location=location)

    def invoke(self, query):
        config = types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            tools=self.tools,
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=query,
            config=config,
        )
        return response