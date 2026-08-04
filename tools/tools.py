from core.tool import Tool


def calculator(expression: str) -> str:
    """Evaluates a basic arithmetic expression, e.g. '12 * (3 + 4)', and returns the result."""
    allowed_chars = "0123456789+-*/(). "
    if not all(char in allowed_chars for char in expression):
        return "error: expression contains unsupported characters"
    return str(eval(expression))


# Every tool the agent can use, name -> Tool instance.
TOOLS = {tool.name: tool for tool in [Tool(calculator)]}