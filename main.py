import os

from core.agent import Agent
from core.memory import Memory
from core.runner import Runner
from prompts.prompt import universal_agent_prompt
from tools import TOOLS
from core.tokenWise import usage_from_response
from data.dataset import load_math_dataset, filter_by_level

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
MODEL = "gemini-2.0-flash"


def build_agent():
    return Agent(
        name="math-agent",
        model=MODEL,
        system_prompt=universal_agent_prompt,
        project=PROJECT_ID,
        tools=TOOLS,
    )


def main():
    dataset = load_math_dataset()
    dataset = filter_by_level(dataset, ["Level 1"])
    problem = dataset[0]["problem"]

    math_agent = build_agent()
    memory = Memory()
    runner = Runner(math_agent, memory)

    response = runner.run(problem)
    usage = usage_from_response(response)

    print("Problem:", problem)
    print("Answer:", response.text)
    print("Token usage:", usage)
    print("Turns in memory:", len(memory.get_history()))


if __name__ == "__main__":
    main()