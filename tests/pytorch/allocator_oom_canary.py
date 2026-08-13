"""Fresh-process diagnostic OOM canary for the PyTorch callback boundary."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import torch

from shadowspill.pytorch.runtime_adapter.abi import AdapterFailure
from shadowspill.pytorch.runtime_adapter.allocator import install_allocator

NO_PROGRESS = 4
REQUEST_BYTES = 128 << 20


def main() -> int:
    installed = install_allocator(
        Path(sys.argv[1]).resolve(),
        device_ordinal=0,
        device_budget_bytes=1 << 30,
        provider_headroom_bytes=512 << 20,
        spill_pool_bytes=1 << 20,
        worker_poll_nanoseconds=10_000,
    )
    impossible = torch.empty((REQUEST_BYTES,), dtype=torch.uint8, device="cuda")
    if impossible.data_ptr() != 0:
        raise AssertionError("failed callback produced a non-null data pointer")
    failure = AdapterFailure()
    status = int(
        installed.library.shadowspill_pytorch_allocator_failure(ctypes.byref(failure))
    )
    if status != NO_PROGRESS or failure.status != NO_PROGRESS:
        raise AssertionError(f"unexpected adapter failure status: {status}")
    if failure.requested_bytes != REQUEST_BYTES:
        raise AssertionError("adapter lost the requested allocation size")
    if failure.runtime.status != NO_PROGRESS:
        raise AssertionError("adapter did not preserve the runtime's first cause")
    if failure.runtime.free_bytes != installed.admission.execution_pool_bytes:
        raise AssertionError("diagnostic free-space accounting is incorrect")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
