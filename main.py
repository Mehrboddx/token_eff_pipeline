import os

from dotenv import load_dotenv

from core.agent import Agent
from core.memory import Memory
from core.runner import Runner
from prompts.prompt import universal_agent_prompt
from tools.tools import PipelineExit, TOOLS

load_dotenv()

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
MODEL = "gemini-2.5-flash"


def build_agent() -> Agent:
    return Agent(
        name="math-agent",
        model=MODEL,
        system_prompt=universal_agent_prompt,
        project=PROJECT_ID,
        tools=TOOLS,
    )


def main() -> None:
    math_agent = build_agent()
    memory = Memory()
    runner = Runner(math_agent, memory)

    print("Math agent ready. Type 'exit' or 'quit' to stop.")

    while True:
        try:
            query = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not query:
            continue

        try:
            response = runner.run(query)
            print("Answer:", response.text)
            print("Turns in memory:", len(memory.get_history()))
            print("Memory content:", memory.get_history())
        except PipelineExit:
            print("Exiting.")
            break
        except Exception as exc:
            print(f"Error while processing the question: {exc}")


if __name__ == "__main__":
    main()