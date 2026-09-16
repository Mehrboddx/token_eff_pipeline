import os
import logging
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

from core.agent import Agent
from core.cpcCompressor import CPCCompressor
from core.geminiCompressor import GeminiCompressor
from core.memory import Memory
from core.runner import Runner
from core.tokenWise import TokenWise
from prompts.prompt import universal_agent_prompt
from tools.tools import PipelineExit, TOOLS

load_dotenv()

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
MODEL = "gemini-2.5-flash"
COMPRESSOR_BACKEND = os.environ.get("COMPRESSOR_BACKEND", "local")  # "local", "gemini", or "cpc"
LOGS_DIR = Path("logs")


def build_context_logger() -> tuple[logging.Logger, Path, str]:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    session_code = uuid4().hex[:8]
    context_log_path = LOGS_DIR / f"model_context_{session_code}.log"

    logger = logging.getLogger(f"model_context_{session_code}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    handler = logging.FileHandler(context_log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    logger.addHandler(handler)

    logger.info("[Session start] code=%s", session_code)

    return logger, context_log_path, session_code


def build_context_monitor(logger: logging.Logger):
    def log_context(history, context_meta=None):
        turn_number = (context_meta or {}).get("turn")
        pass_number = (context_meta or {}).get("pass")
        mode = (context_meta or {}).get("mode")
        logger.info("[Turn %s | pass %s start] mode=%s", turn_number, pass_number, mode)
        for index, item in enumerate(history, start=1):
            logger.info("%s. %s (%s): %s", index, item["role"], item.get("kind"), item["text"])
        logger.info("[Turn %s | pass %s end]", turn_number, pass_number)

    return log_context


def write_turn_separator(logger: logging.Logger) -> None:
    for handler in logger.handlers:
        if hasattr(handler, "stream") and handler.stream is not None:
            handler.stream.write("\n")
            handler.flush()


def close_context_logger(logger: logging.Logger, session_code: str) -> None:
    logger.info("[Session end] code=%s", session_code)
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def build_agent(context_monitor) -> Agent:
    return Agent(
        name="math-agent",
        model=MODEL,
        system_prompt=universal_agent_prompt,
        project=PROJECT_ID,
        tools=TOOLS,
        context_monitor=context_monitor,
    )


def build_tokenwise() -> TokenWise:
    if COMPRESSOR_BACKEND == "gemini":
        return TokenWise(model=GeminiCompressor(project=PROJECT_ID, location=LOCATION, model=MODEL))

    if COMPRESSOR_BACKEND == "cpc":
        return TokenWise(model=CPCCompressor())

    try:
        # Without a loaded CPC tokenizer, TokenWise falls back to counting
        # words, not tokens — token_budget then means something quite
        # different from what gets sent to Gemini. tiktoken gives a much
        # closer real token estimate for the same word count.
        return TokenWise(use_openai_tokenizer=True)
    except ImportError:
        return TokenWise()


def main() -> None:
    logger, context_log_path, session_code = build_context_logger()
    math_agent = build_agent(build_context_monitor(logger))
    memory = Memory()
    tokenwise = build_tokenwise()
    # compression_sentence_threshold/compression_token_budget/recent_turns
    # now just take Runner's defaults (see core/runner.py), which are the
    # values validated against eval/longmemeval.py rather than the small
    # test values this used to hardcode.
    runner = Runner(math_agent, memory, tokenwise=tokenwise)
    logger.info("[Compressor] backend=%s", COMPRESSOR_BACKEND)

    print("Math agent ready. Type 'exit' or 'quit' to stop.")
    print(f"Session code: {session_code}")
    print(f"Compressor backend: {COMPRESSOR_BACKEND}")
    print(f"Model context will be logged to {context_log_path}")

    try:
        turn_index = 0
        while True:
            try:
                query = input("\nQuestion: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if not query:
                continue

            try:
                turn_index += 1
                runner.current_turn = turn_index
                logger.info("[Turn %s start]", turn_index)
                response = runner.run(query)
                print("Answer:", response.text)
                print("Turns in memory:", len(memory.get_history()))
                print("Memory content:", memory.get_history())
                logger.info("[Turn %s end]", turn_index)
                write_turn_separator(logger)
            except PipelineExit:
                print("Exiting.")
                break
            except Exception as exc:
                print(f"Error while processing the question: {exc}")
    finally:
        close_context_logger(logger, session_code)


if __name__ == "__main__":
    main()