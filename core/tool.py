from google.genai import types


class Tool:
    """Wraps a plain Python function as an agent tool. The declaration the
    model sees is generated automatically from the function's type hints and
    docstring, so a tool only has to be written once."""

    def __init__(self, function):
        self.function = function
        self.name = function.__name__
        self.declaration = types.FunctionDeclaration.from_callable_with_api_option(
            callable=function,
            api_option="VERTEX_AI",
        )

    def run(self, **kwargs):
        return self.function(**kwargs)