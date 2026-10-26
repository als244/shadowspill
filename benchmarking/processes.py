"""Fresh Python workers; distributed workers share the same logical CUDA ordinal."""

import sys


def worker_command(module, arguments, *, ranks=1):
    python = [sys.executable, *(["-S"] if sys.flags.no_site else [])]
    if ranks == 1:
        return [*python, "-u", "-m", module, *map(str, arguments)]
    return [
        *python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={ranks}",
        "--no-python",
        *python,
        "-u",
        "-m",
        "benchmarking.device_worker",
        module,
        *map(str, arguments),
    ]
