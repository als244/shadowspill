"""Fresh-process canary proving bad CUDA kernels retain their backend error."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

from shadowspill.pytorch.runtime_adapter.abi import AdapterFailure
from shadowspill.pytorch.runtime_adapter.allocator import install_allocator
from shadowspill.pytorch.runtime_adapter.failures import read_allocator_failure
from tests.integration.pytorch.runtime_helpers import two_pool_topology


@triton.jit
def _invalid_store_kernel(value: tl.tensor, invalid_offset: tl.constexpr) -> None:
    """Store far outside the slab so CUDA reports an illegal memory access."""

    tl.store(value + invalid_offset, 1.0)


def main() -> int:
    installed = install_allocator(
        Path(sys.argv[1]).resolve(),
        device_ordinal=0,
        device_budget_bytes=2 << 30,
        provider_headroom_bytes=512 << 20,
        **two_pool_topology(1 << 20),
        worker_poll_nanoseconds=1_000,
    )
    value = torch.zeros(1, dtype=torch.float32, device="cuda")
    _invalid_store_kernel[(1,)](value, invalid_offset=1 << 42)
    try:
        torch.cuda.synchronize()
    except RuntimeError as error:
        message = str(error).lower()
        if "illegal memory access" not in message:
            raise AssertionError(f"unexpected CUDA error: {error}") from error
        if read_allocator_failure(
            installed.library,
            "execute intentionally invalid CUDA kernel",
        ) is not None:
            raise AssertionError(
                "bad CUDA kernel was mislabeled as allocator OOM"
            ) from error
        failure = AdapterFailure()
        status = int(
            installed.library.shadowspill_pytorch_allocator_failure(
                ctypes.byref(failure)
            )
        )
        if status != 0:
            raise AssertionError(
                f"bad CUDA kernel latched unrelated runtime status {status}"
            ) from error
        return 0
    raise AssertionError("invalid CUDA store unexpectedly completed")


if __name__ == "__main__":
    raise SystemExit(main())
