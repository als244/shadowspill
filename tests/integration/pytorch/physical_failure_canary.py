"""Fresh-process external-provider physical-growth failure canary."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

from shadowspill.pytorch.runtime_adapter.abi import AdapterFailure, AdapterStatistics
from shadowspill.pytorch.runtime_adapter.allocator import install_allocator
from shadowspill.status import Status
from tests.integration.pytorch.runtime_helpers import two_pool_topology

MIB = 1 << 20
PLAN_VIOLATION = Status.PLAN_VIOLATION


def main() -> int:
    installed = install_allocator(
        Path(sys.argv[1]).resolve(),
        device_ordinal=0,
        device_budget_bytes=1 << 30,
        provider_headroom_bytes=64 * MIB,
        **two_pool_topology(1 * MIB),
        worker_poll_nanoseconds=10_000,
    )
    library = installed.library
    if int(library.shadowspill_pytorch_seal_physical_budget(64 * MIB, 16)) != 0:
        raise AssertionError("physical budget did not seal before growth injection")

    cuda = ctypes.CDLL("libcuda.so.1")
    cuda.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
    cuda.cuMemAlloc_v2.restype = ctypes.c_int
    cuda.cuMemFree_v2.argtypes = [ctypes.c_uint64]
    cuda.cuMemFree_v2.restype = ctypes.c_int
    external = ctypes.c_uint64()
    if cuda.cuMemAlloc_v2(ctypes.byref(external), 80 * MIB) != 0:
        raise AssertionError("failed to inject external provider growth")
    try:
        status = int(library.shadowspill_pytorch_check_physical_budget())
        if status != PLAN_VIOLATION:
            raise AssertionError(f"unexpected physical-check status {status}")
        failure = AdapterFailure()
        if (
            int(library.shadowspill_pytorch_allocator_failure(ctypes.byref(failure)))
            != PLAN_VIOLATION
        ):
            raise AssertionError("physical violation was not latched")
        statistics = AdapterStatistics()
        if (
            int(
                library.shadowspill_pytorch_allocator_statistics(
                    ctypes.byref(statistics)
                )
            )
            != 0
        ):
            raise AssertionError("statistics query failed")
        if statistics.callback_failures != 0:
            raise AssertionError(
                "provider growth was misclassified as callback failure"
            )
        if statistics.observed_external_high_water_bytes <= 64 * MIB:
            raise AssertionError("external high-water did not exceed its reservation")
        if statistics.peak_process_physical_bytes <= 1 << 30:
            raise AssertionError("negative canary did not actually exceed the cap")
    finally:
        if cuda.cuMemFree_v2(external.value) != 0:
            raise AssertionError("failed to release injected provider allocation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
