"""Launch a single-node DP worker with its assigned GPU exposed as cuda:0.

Run under torchrun, passing the target module and its arguments. Stable logical
ordinals let ranks share compiler identities while retaining global DP ranks.
"""

import os
import runpy
import sys


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: logical_device_worker MODULE [ARGUMENT ...]")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = visible.split(",") if visible else None
    if local_rank < 0 or (devices is not None and local_rank >= len(devices)):
        raise ValueError("worker GPU index is outside CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = (
        devices[local_rank] if devices is not None else str(local_rank)
    )
    os.environ["LOCAL_RANK"] = "0"
    module = sys.argv[1]
    sys.argv = [module, *sys.argv[2:]]
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
