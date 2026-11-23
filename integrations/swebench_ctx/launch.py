"""`sweagent run-batch` with the CPC recall history processor registered.

SWE-agent picks history processors from a pydantic union, so CPCRecall is added
to that union before the agent configuration is imported. The fork's own code
(and its masking, summary and raw policies) stays untouched.

    python -m integrations.swebench_ctx.launch <sweagent run-batch arguments>
"""

import sys
from typing import Annotated, Union, get_args

from pydantic import Field


def register() -> None:
    import sweagent.agent.history_processors as processors

    from integrations.swebench_ctx.recall import CPCRecall

    if "sweagent.agent.agents" in sys.modules:
        raise RuntimeError("register() must run before sweagent.agent.agents is imported")
    union = get_args(processors.HistoryProcessor)[0]
    members = tuple(get_args(union)) + (CPCRecall,)
    processors.HistoryProcessor = Annotated[Union[members], Field(discriminator="type")]  # noqa: UP007
    processors.CPCRecall = CPCRecall


def main() -> None:
    register()
    from sweagent.run.run_batch import run_from_cli

    run_from_cli(sys.argv[1:])


if __name__ == "__main__":
    main()
