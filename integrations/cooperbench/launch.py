"""CooperBench's own CLI, with the CPC-Mem agent installed first.

Takes exactly the arguments of `cooperbench` (run / eval / ...):

    python -m integrations.cooperbench.launch run -n flash_full -s flash \\
        --agent-config integrations/cooperbench/arms/full.yaml ...
"""

import sys

from integrations.cooperbench.agent import install


def main() -> None:
    install()
    from cooperbench.cli import main as cooperbench_main

    sys.argv[0] = "cooperbench"
    cooperbench_main()


if __name__ == "__main__":
    main()
