"""Cold arena sizing leaves room for mature external process memory."""

import ctypes
import json
import sys
from pathlib import Path

from shadowspill.memory import DevicePool, PinnedHostPool, TransferRoute
from shadowspill.pytorch.frontend import PyTorchFrontend
from shadowspill.runtime import Runtime
from shadowspill.runtime.abi import AdapterStatistics
from shadowspill.runtime.budget import check_physical_budget


def main():
    floor = 2 << 30
    headroom = 256 << 20
    # The usable tail is real capacity; it must not disappear through an
    # assumed 2 MiB physical-page rounding before Plan admission.
    budget = (4 << 30) + (512 << 10)
    runtime = Runtime(
        frontend=PyTorchFrontend(),
        library_path=Path(sys.argv[1]),
        pools={
            "execution": DevicePool(
                budget,
                external_baseline=floor,
                external_headroom=headroom,
                reject_overbudget=True,
            ),
            "spill": PinnedHostPool(64 << 20),
        },
        routes={
            "fetch": TransferRoute("spill", "execution"),
            "evict": TransferRoute("execution", "spill"),
        },
        calibrate=False,
    )
    installed = runtime._installed
    admission = installed.admission
    assert admission.baseline_bytes < floor
    assert admission.reserved_baseline_bytes == floor
    assert admission.external_headroom_bytes == headroom
    assert admission.allocator_pool_bytes == budget - floor - headroom
    check_physical_budget(installed)
    cuda = ctypes.CDLL("libcuda.so.1")
    cuda.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
    cuda.cuMemAlloc_v2.restype = ctypes.c_int
    cuda.cuMemFree_v2.argtypes = [ctypes.c_uint64]
    cuda.cuMemFree_v2.restype = ctypes.c_int
    allocation = ctypes.c_uint64()
    assert cuda.cuMemAlloc_v2(ctypes.byref(allocation), 512 << 20) == 0
    try:
        # This exceeds the growth headroom alone, but fits the mature baseline
        # reserved before the arena was allocated. It must pass strict admission.
        check_physical_budget(installed)
        statistics = AdapterStatistics()
        assert (
            installed.library.shadowspill_pytorch_allocator_statistics(
                ctypes.byref(statistics)
            )
            == 0
        )
        assert statistics.observed_external_high_water_bytes > headroom
        assert statistics.peak_process_physical_bytes <= budget
        print(
            json.dumps(
                dict(
                    observed_baseline_bytes=int(admission.baseline_bytes),
                    reserved_baseline_bytes=int(admission.reserved_baseline_bytes),
                    headroom_bytes=headroom,
                    allocator_pool_bytes=int(admission.allocator_pool_bytes),
                    raw_external_growth_bytes=int(
                        statistics.observed_external_high_water_bytes
                    ),
                    peak_process_bytes=int(statistics.peak_process_physical_bytes),
                    strict_check_passed=True,
                )
            ),
            flush=True,
        )
    finally:
        assert cuda.cuMemFree_v2(allocation.value) == 0
        runtime.close()


if __name__ == "__main__":
    main()
