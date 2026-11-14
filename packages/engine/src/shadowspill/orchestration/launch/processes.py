"""Fresh Python workers for repository clients, with one visible GPU per rank."""

import sys
from collections.abc import Iterable


def worker_command(
    module: str, arguments: Iterable[object], *, ranks: int = 1
) -> list[str]:
    python = [sys.executable, *(["-S"] if sys.flags.no_site else [])]
    if ranks == 1:
        return [*python, "-u", "-m", module, *map(str, arguments)]
    return distributed_command(
        [
            *python,
            "-u",
            "-m",
            "shadowspill.orchestration.launch.rank_device",
            "-m",
            module,
            *map(str, arguments),
        ],
        ranks=ranks,
    )


def distributed_command(command: Iterable[str], *, ranks: int) -> list[str]:
    """Launch an already wrapped per-rank command on one node."""
    if ranks == 1:
        return list(command)
    python = [sys.executable, *(["-S"] if sys.flags.no_site else [])]
    return [
        *python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={ranks}",
        "--no-python",
        *command,
    ]
