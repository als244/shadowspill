from __future__ import annotations

import ctypes
from pathlib import Path

import pytest

from shadowspill.pytorch._abi import (
    AdapterCapabilities,
    AdapterConfig,
    AdapterFailure,
    AdapterStatistics,
    Allocation,
    AllocationEvent,
    CudaStatistics,
    ObjectBinding,
    ObjectSnapshot,
    ObjectUpdate,
    PhysicalAdmission,
    PhysicalMemory,
    RuntimeAction,
    RuntimeFailure,
    RuntimeStatistics,
    TaskHostTiming,
    TraceConfig,
    TraceEvent,
    TraceSummary,
    configure_adapter_library,
)
from shadowspill.pytorch._allocator import (
    AllocatorInstallError,
    InstalledAllocator,
    _function_pointer,
    install_allocator,
    resize_host_arena,
)


class _Function:
    argtypes: object = None
    restype: object = None


class _Library:
    shadowspill_pytorch_adapter_capabilities = _Function()
    shadowspill_pytorch_physical_admission = _Function()
    shadowspill_pytorch_physical_memory = _Function()
    shadowspill_pytorch_seal_physical_budget = _Function()
    shadowspill_pytorch_check_physical_budget = _Function()
    shadowspill_pytorch_allocator_bootstrap = _Function()
    shadowspill_pytorch_allocator_statistics = _Function()
    shadowspill_pytorch_allocator_failure = _Function()
    shadowspill_pytorch_allocator_wait_idle = _Function()
    shadowspill_pytorch_debug_task_timing_enable = _Function()
    shadowspill_pytorch_debug_task_timing_read = _Function()
    shadowspill_pytorch_debug_task_timing_disable = _Function()
    shadowspill_pytorch_resize_host_arena = _Function()
    shadowspill_pytorch_allocation_telemetry_start = _Function()
    shadowspill_pytorch_allocation_telemetry_stop = _Function()
    shadowspill_pytorch_allocation_telemetry_read = _Function()
    shadowspill_pytorch_trace_prepare = _Function()
    shadowspill_pytorch_trace_begin = _Function()
    shadowspill_pytorch_trace_end = _Function()
    shadowspill_pytorch_trace_read = _Function()
    shadowspill_pytorch_allocation_for_pointer = _Function()
    shadowspill_pytorch_register_host_object = _Function()
    shadowspill_pytorch_write_host_object = _Function()
    shadowspill_pytorch_read_host_object = _Function()
    shadowspill_pytorch_unregister_object = _Function()
    shadowspill_pytorch_bind_registered_allocation = _Function()
    shadowspill_pytorch_transfer_output_to_caller = _Function()
    shadowspill_pytorch_promote_allocation = _Function()
    shadowspill_pytorch_before_task = _Function()
    shadowspill_pytorch_after_task = _Function()
    shadowspill_pytorch_object_snapshot = _Function()
    shadowspill_pytorch_abort_task_range = _Function()


def test_declarative_adapter_abi_has_expected_c_layout() -> None:
    assert ctypes.sizeof(AdapterConfig) == 40
    assert ctypes.sizeof(AdapterCapabilities) == 20
    assert ctypes.sizeof(RuntimeStatistics) == 24 * 8
    assert ctypes.sizeof(AllocationEvent) == 64
    assert ctypes.sizeof(Allocation) == 40
    assert ctypes.sizeof(CudaStatistics) == 22 * 8
    assert ctypes.sizeof(RuntimeFailure) == 48
    assert ctypes.sizeof(AdapterFailure) == 72
    assert ctypes.sizeof(AdapterStatistics) == 448
    assert ctypes.sizeof(ObjectBinding) == 40
    assert ctypes.sizeof(ObjectUpdate) == 16
    assert ctypes.sizeof(RuntimeAction) == 16
    assert ctypes.sizeof(ObjectSnapshot) == 88
    assert ctypes.sizeof(PhysicalAdmission) == 72
    assert ctypes.sizeof(PhysicalMemory) == 32
    assert ctypes.sizeof(TaskHostTiming) == 88
    assert ctypes.sizeof(TraceConfig) == 24
    assert ctypes.sizeof(TraceEvent) == 80
    assert ctypes.sizeof(TraceSummary) == 72


def test_adapter_signatures_are_configured_together() -> None:
    library = _Library()
    configure_adapter_library(library)
    assert library.shadowspill_pytorch_adapter_capabilities.restype is ctypes.c_uint32
    assert library.shadowspill_pytorch_physical_admission.argtypes == [
        ctypes.POINTER(PhysicalAdmission)
    ]
    assert library.shadowspill_pytorch_physical_memory.argtypes == [
        ctypes.POINTER(PhysicalMemory)
    ]
    assert library.shadowspill_pytorch_seal_physical_budget.argtypes == [
        ctypes.c_uint64,
        ctypes.c_uint64,
    ]
    assert library.shadowspill_pytorch_check_physical_budget.argtypes == []
    assert library.shadowspill_pytorch_allocator_bootstrap.argtypes == [
        ctypes.POINTER(AdapterConfig)
    ]
    assert library.shadowspill_pytorch_allocator_statistics.argtypes == [
        ctypes.POINTER(AdapterStatistics)
    ]
    assert library.shadowspill_pytorch_allocator_failure.argtypes == [
        ctypes.POINTER(AdapterFailure)
    ]
    assert library.shadowspill_pytorch_allocator_wait_idle.argtypes == []
    assert library.shadowspill_pytorch_debug_task_timing_enable.argtypes == [
        ctypes.c_uint32
    ]
    assert library.shadowspill_pytorch_debug_task_timing_read.argtypes == [
        ctypes.POINTER(TaskHostTiming),
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    assert library.shadowspill_pytorch_debug_task_timing_disable.argtypes == []
    assert library.shadowspill_pytorch_resize_host_arena.argtypes == [ctypes.c_uint64]
    assert library.shadowspill_pytorch_allocation_telemetry_start.argtypes == [
        ctypes.c_uint64
    ]
    assert library.shadowspill_pytorch_allocation_telemetry_stop.argtypes == []
    assert library.shadowspill_pytorch_allocation_telemetry_read.argtypes == [
        ctypes.POINTER(AllocationEvent),
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    assert library.shadowspill_pytorch_trace_prepare.argtypes == [
        ctypes.POINTER(TraceConfig)
    ]
    assert library.shadowspill_pytorch_trace_begin.argtypes == [ctypes.c_uint64]
    assert library.shadowspill_pytorch_trace_end.argtypes == []
    assert library.shadowspill_pytorch_trace_read.argtypes == [
        ctypes.POINTER(TraceSummary),
        ctypes.POINTER(TraceEvent),
        ctypes.c_uint64,
        ctypes.POINTER(AllocationEvent),
        ctypes.c_uint64,
    ]
    assert library.shadowspill_pytorch_allocation_for_pointer.argtypes == [
        ctypes.c_uint64,
        ctypes.POINTER(Allocation),
    ]
    assert library.shadowspill_pytorch_register_host_object.argtypes == [
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint8,
        ctypes.c_uint64,
    ]
    assert library.shadowspill_pytorch_write_host_object.argtypes == [
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
    ]
    assert library.shadowspill_pytorch_read_host_object.argtypes == [
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
    ]
    assert library.shadowspill_pytorch_unregister_object.argtypes == [ctypes.c_uint64]
    assert library.shadowspill_pytorch_bind_registered_allocation.argtypes == [
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.POINTER(ObjectBinding),
    ]
    assert library.shadowspill_pytorch_transfer_output_to_caller.argtypes == [
        ctypes.c_uint64,
        ctypes.POINTER(Allocation),
    ]
    assert library.shadowspill_pytorch_promote_allocation.argtypes == [
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.POINTER(ObjectBinding),
    ]
    assert library.shadowspill_pytorch_before_task.argtypes == [
        ctypes.c_uint64,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_uint32,
        ctypes.POINTER(ObjectBinding),
        ctypes.c_uint32,
    ]
    assert library.shadowspill_pytorch_after_task.argtypes == [
        ctypes.c_uint64,
        ctypes.c_size_t,
        ctypes.POINTER(ObjectUpdate),
        ctypes.c_uint32,
        ctypes.POINTER(RuntimeAction),
        ctypes.c_uint32,
    ]
    assert library.shadowspill_pytorch_abort_task_range.argtypes == []


def test_planning_host_growth_updates_admission_and_enforces_overlap() -> None:
    class _ResizeLibrary:
        host_bytes = 16

        def shadowspill_pytorch_resize_host_arena(self, value: int) -> int:
            self.host_bytes = value
            return 0

        def shadowspill_pytorch_physical_admission(self, output: object) -> int:
            admission = ctypes.cast(output, ctypes.POINTER(PhysicalAdmission))[0]
            admission.host_arena_bytes = self.host_bytes
            return 0

    admission = PhysicalAdmission()
    admission.host_arena_bytes = 16
    installed = InstalledAllocator(
        library=_ResizeLibrary(),
        allocator=object(),
        path=Path("/adapter"),
        admission=admission,
    )
    resize_host_arena(installed, host_arena_bytes=32, host_budget_bytes=64)
    assert installed.admission.host_arena_bytes == 32
    resize_host_arena(installed, host_arena_bytes=32, host_budget_bytes=64)
    with pytest.raises(AllocatorInstallError, match="shrink"):
        resize_host_arena(installed, host_arena_bytes=31, host_budget_bytes=64)
    with pytest.raises(AllocatorInstallError, match="exceeds"):
        resize_host_arena(installed, host_arena_bytes=40, host_budget_bytes=64)


def test_missing_callback_symbol_has_field_specific_error() -> None:
    with pytest.raises(AllocatorInstallError, match="missing_callback"):
        _function_pointer(object(), "missing_callback")


def test_installer_rejects_missing_library(tmp_path: Path) -> None:
    missing = tmp_path / "libshadowspill_pytorch.so"
    with pytest.raises(AllocatorInstallError, match="does not exist"):
        install_allocator(
            missing,
            device_ordinal=0,
            device_budget_bytes=1,
            provider_headroom_bytes=0,
            host_arena_bytes=0,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"device_ordinal": -1}, "ordinal"),
        ({"device_budget_bytes": 0}, "budget"),
        ({"provider_headroom_bytes": -1}, "headroom"),
        ({"provider_headroom_bytes": 1024}, "headroom"),
        ({"host_arena_bytes": -1}, "host arena"),
        ({"progress_poll_nanoseconds": -1}, "poll"),
    ],
)
def test_installer_rejects_invalid_physical_configuration(
    tmp_path: Path, overrides: dict[str, int], message: str
) -> None:
    arguments = {
        "device_ordinal": 0,
        "device_budget_bytes": 1024,
        "provider_headroom_bytes": 0,
        "host_arena_bytes": 0,
        "progress_poll_nanoseconds": 0,
    }
    arguments.update(overrides)
    with pytest.raises(AllocatorInstallError, match=message):
        install_allocator(tmp_path / "missing.so", **arguments)
