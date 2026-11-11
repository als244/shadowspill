"""Fresh Python workers for repository clients, with one visible GPU per rank."""

import sys


def worker_command(module, arguments, *, ranks=1):
    python = [sys.executable, *(["-S"] if sys.flags.no_site else [])]
    if ranks == 1:
        return [*python, "-u", "-m", module, *map(str, arguments)]
    return distributed_command(
        [
            *python,
            "-u",
            "-m",
            "orchestration.rank_device",
            "-m",
            module,
            *map(str, arguments),
        ],
        ranks=ranks,
    )


def distributed_command(command, *, ranks):
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
