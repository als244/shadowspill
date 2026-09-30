"""Fresh-process external-memory growth failure canary."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

from shadowspill.pytorch.frontend import PyTorchFrontend
from shadowspill.runtime.abi import AdapterFailure, AdapterStatistics
from shadowspill.runtime.bootstrap import install_runtime
from shadowspill.status import Status
from tests.integration.pytorch.runtime_helpers import two_pool_topology

MIB = 1 << 20
PLAN_VIOLATION = Status.PLAN_VIOLATION


def main(*, report_only: bool = False, headroom_mib: int = 256) -> int:
    installed = install_runtime(
        Path(sys.argv[1]).resolve(),
        frontend=PyTorchFrontend(),
        device_ordinal=0,
        device_budget_bytes=2 << 30,
        external_headroom_bytes=headroom_mib * MIB,
        reject_overbudget=not report_only,
        **two_pool_topology(1 * MIB),
        worker_poll_nanoseconds=10_000,
    )
    library = installed.library
    if int(library.shadowspill_pytorch_seal_physical_budget(256 * MIB, 16)) != 0:
        raise AssertionError("physical budget did not seal before growth injection")

    cuda = ctypes.CDLL("libcuda.so.1")
    cuda.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
    cuda.cuMemAlloc_v2.restype = ctypes.c_int
    cuda.cuMemFree_v2.argtypes = [ctypes.c_uint64]
    cuda.cuMemFree_v2.restype = ctypes.c_int
    external = ctypes.c_uint64()
    if cuda.cuMemAlloc_v2(ctypes.byref(external), 320 * MIB) != 0:
        raise AssertionError("failed to inject external memory growth")
    try:
        status = int(library.shadowspill_pytorch_check_physical_budget())
        expected = Status.OK if report_only else PLAN_VIOLATION
        if status != expected:
            raise AssertionError(f"unexpected physical-check status {status}")
        failure = AdapterFailure()
        if (
            int(library.shadowspill_pytorch_allocator_failure(ctypes.byref(failure)))
            != expected
        ):
            raise AssertionError(
                "physical failure latch disagrees with enforcement mode"
            )
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
                "external memory growth was misclassified as callback failure"
            )
        if statistics.observed_external_high_water_bytes <= 256 * MIB:
            raise AssertionError("external high-water did not exceed its reservation")
        if statistics.peak_process_physical_bytes <= 2 << 30:
            raise AssertionError("negative canary did not actually exceed the cap")
        if report_only:
            # Sealing repeats the physical check. The flag must remain report-only
            # after actual external growth, including on later checks.
            if (
                int(library.shadowspill_pytorch_seal_physical_budget(384 * MIB, 16))
                != 0
            ):
                raise AssertionError("report-only sealing rejected external growth")
            if int(library.shadowspill_pytorch_check_physical_budget()) != 0:
                raise AssertionError(
                    "report-only repeated check rejected external growth"
                )
    finally:
        if cuda.cuMemFree_v2(external.value) != 0:
            raise AssertionError("failed to release injected external allocation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
