"""MemoryArena's own code (third_party/MemoryArena, pinned), imported without its package
__init__ files: those import every environment and memory system (WebShop, Mirix, Mem0,
Letta, torch, ...), which the formal-reasoning tasks do not use. Each directory is
registered as a bare package, so only the modules named here load."""

import os
import sys
import types
from pathlib import Path

DEFAULT_REPO = Path(__file__).resolve().parents[2] / "third_party" / "MemoryArena"


def _bare_package(name: str, directory: Path) -> None:
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(directory)]
        sys.modules[name] = package


def load(repo: Path | str | None = None) -> types.SimpleNamespace:
    repo = Path(repo or os.environ.get("MEMORYARENA_REPO") or DEFAULT_REPO)
    if not (repo / "agent" / "math.py").is_file():
        raise FileNotFoundError(f"MemoryArena not found at {repo} (scripts/memoryarena.sh setup)")
    _bare_package("agent", repo / "agent")
    _bare_package("env_systems", repo / "env" / "env_systems")
    _bare_package("memory_systems", repo / "memory" / "memory_systems")
    # agent/math.py imports formal_reasoning_env.llm_backend from env/env_systems on sys.path.
    sys.path.insert(0, str(repo / "env" / "env_systems"))
    from agent.math import MathAgent
    from env_systems.math_env import MathEnvironment
    from memory_systems.long_context import LongContextMemorySystem
    from memory_systems.rag import RAGMemorySystem

    return types.SimpleNamespace(MathAgent=MathAgent, MathEnvironment=MathEnvironment,
                                 LongContextMemorySystem=LongContextMemorySystem, RAGMemorySystem=RAGMemorySystem)


def load_travel(repo: Path | str | None = None) -> types.SimpleNamespace:
    """The group-travel pieces: their agent (ReAct over local CSV tools), their run_travel.py
    helpers (plan formatting and parsing), data conversion and evaluation functions."""
    repo = Path(repo or os.environ.get("MEMORYARENA_REPO") or DEFAULT_REPO)
    flights = repo / "env" / "env_systems" / "travel_planner_env" / "database" / "flights" / "clean_Flights_2022.csv"
    if not flights.is_file():
        raise FileNotFoundError(f"{flights} missing (scripts/memoryarena.sh setup-travel)")
    _bare_package("agent", repo / "agent")
    _bare_package("env", repo / "env")
    _bare_package("env.env_systems", repo / "env" / "env_systems")
    _bare_package("memory_systems", repo / "memory" / "memory_systems")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))  # run_travel.py is a top-level script there
    import agent.travel_planner as agent_module
    import run_travel as reference
    from env.env_systems.travel_planner_env import eval as evaluation
    from env.env_systems.travel_planner_env.combination import parse_plan_text
    from env.env_systems.travel_planner_env.data_loader import _convert_row
    from memory_systems.long_context import LongContextMemorySystem
    from memory_systems.rag import RAGMemorySystem

    return types.SimpleNamespace(agent_module=agent_module, TravelPlannerAgent=agent_module.TravelPlannerAgent,
                                 format_person_plan=reference.format_person_plan,
                                 parse_all_plans=reference.parse_all_plans, parse_plan_text=parse_plan_text,
                                 convert_row=_convert_row, evaluation=evaluation,
                                 LongContextMemorySystem=LongContextMemorySystem, RAGMemorySystem=RAGMemorySystem)
