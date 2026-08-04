from pydantic import BaseModel
from google import genai

class agent():
    def __init__(self, name, model, system_prompt, project_id, location = "global", tools = None):
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.project_id = project_id
        self.location = location
        self.tools = tools
        self.client = genai.Client(vertexai=True, project_id=self.project_id, location = self.location)

    def invoke(self, query):
        response  = self.client.generate_content(model = self.model, prompt = self.system_prompt + query)
        return response