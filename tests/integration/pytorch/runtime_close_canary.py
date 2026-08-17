"""Verify deterministic runtime teardown and the closed allocator shim."""

from __future__ import annotations

import torch

from shadowspill.memory import device, pinned_host, transfer_route
from shadowspill.pytorch import Runtime


def main() -> int:
    runtime = Runtime(
        pools={
            "execution": device(
                physical_capacity=2 << 30,
                provider_headroom=512 << 20,
            ),
            "spill": pinned_host(capacity=256 << 20),
        },
        routes={
            "fetch": transfer_route(source="spill", destination="execution"),
            "evict": transfer_route(source="execution", destination="spill"),
        },
        calibrate=False,
    )

    runtime.close()
    runtime.close()

    try:
        torch.empty((1024,), device="cuda")
    except RuntimeError as error:
        message = str(error)
        if "status: 9 (runtime is closed)" not in message:
            raise AssertionError(message) from error
        if "canonical_task:" in message:
            raise AssertionError(
                "a post-close allocation was incorrectly attributed to a task"
            ) from error
    else:
        raise AssertionError("the closed allocator accepted a device allocation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
