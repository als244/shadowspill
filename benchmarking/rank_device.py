"""Launch an entrypoint with one GPU visible per distributed process.

Run before importing CUDA-dependent modules, for example::

    torchrun --standalone --nproc-per-node=8 -m benchmarking.rank_device \
        -m benchmarking.quickstart --distributed ...

Global/local rank identities are preserved. ShadowSpill's automatic device
selection uses ordinal zero when only one GPU is visible. This gives every
rank the same compiler device label while retaining distinct physical GPUs.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from collections.abc import MutableMapping, Sequence


def isolate_device(environment: MutableMapping[str, str]) -> str:
    """Select from the launcher's GPU mask without importing CUDA libraries."""
    local_rank = int(environment.get("LOCAL_RANK", "0"))
    if local_rank < 0:
        raise ValueError("LOCAL_RANK must be nonnegative")
    original = environment.get("CUDA_VISIBLE_DEVICES")
    if original is None:
        selected = str(local_rank)
    else:
        visible = [item.strip() for item in original.split(",")]
        if any(not item or item == "-1" for item in visible):
            raise ValueError("CUDA_VISIBLE_DEVICES does not expose a usable GPU")
        # A scheduler may already bind each process to a single distinct GPU.
        if len(visible) == 1:
            selected = visible[0]
        elif local_rank >= len(visible):
            raise ValueError("LOCAL_RANK exceeds the launcher's visible GPU count")
        else:
            selected = visible[local_rank]
        environment["SHADOWSPILL_ORIGINAL_CUDA_VISIBLE_DEVICES"] = original
    environment["CUDA_VISIBLE_DEVICES"] = selected
    return selected


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--module", action="store_true")
    parser.add_argument("entrypoint", help="Python module with -m, otherwise a script")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        isolate_device(os.environ)
    except ValueError as error:
        parser.error(str(error))
    sys.argv = [args.entrypoint, *args.arguments]
    if args.module:
        runpy.run_module(args.entrypoint, run_name="__main__", alter_sys=True)
    else:
        runpy.run_path(args.entrypoint, run_name="__main__")


if __name__ == "__main__":
    main()
