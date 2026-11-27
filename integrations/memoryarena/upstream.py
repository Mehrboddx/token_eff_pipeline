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
