from __future__ import annotations

import ctypes
from pathlib import Path

import pytest

from shadowspill.pytorch.runtime_adapter import failures as failures_module
from shadowspill.pytorch.runtime_adapter.abi import (
    AdapterCapabilities,
    AdapterConfig,
    AdapterFailure,
    AdapterStatistics,
    Allocation,
    AllocationEvent,
    BackendStatistics,
    FixedDependencyDescription,
    FixedLayoutDescription,
    FixedPlacementDescription,
    ObjectBinding,
    ObjectDescription,
    ObjectLocationSnapshot,
    ObjectSnapshot,
    ObjectUpdate,
    PhysicalAdmission,
    PhysicalMemory,
    RuntimeAction,
    RuntimeFailure,
    RuntimeStatistics,
    TaskDispatchTiming,
    TraceConfig,
    TraceEvent,
    TraceSummary,
    TransferCalibrationConfig,
    TransferProfile,
    configure_adapter_library,
    configure_runtime_library,
)
from shadowspill.pytorch.runtime_adapter.allocator import (
    AllocatorInstallError,
    InstalledAllocator,
    PoolBootstrap,
    RouteBootstrap,
    _function_pointer,
    install_allocator,
    validate_dynamic_execution_reservation,
)


def _two_pool_topology(spill_bytes: int = 1) -> dict[str, object]:
    return {
        "allocator_pool_id": 0,
        "pools": (
            PoolBootstrap(0, 0, 0),
            PoolBootstrap(1, 1, spill_bytes),
        ),
        "routes": (
            RouteBootstrap(0, "fetch", 1, 0),
            RouteBootstrap(1, "evict", 0, 1),
        ),
    }


class _IdleRuntime:
    """The neutral runtime as a fake library whose drain always succeeds."""

    @staticmethod
    def shadowspill_runtime_wait_idle(runtime_handle: int) -> int:
        del runtime_handle
        return 0


class _Function:
    argtypes: object = None
    restype: object = None


class _Library:
    shadowspill_pytorch_adapter_capabilities = _Function()
    shadowspill_pytorch_runtime_handle = _Function()
    shadowspill_pytorch_profile_range_begin = _Function()
    shadowspill_pytorch_profile_range_end = _Function()
    shadowspill_pytorch_physical_admission = _Function()
    shadowspill_pytorch_physical_memory = _Function()
    shadowspill_pytorch_seal_physical_budget = _Function()
    shadowspill_pytorch_check_physical_budget = _Function()
    shadowspill_pytorch_allocator_bootstrap = _Function()
    shadowspill_pytorch_allocator_close = _Function()
    shadowspill_pytorch_allocator_statistics = _Function()
    shadowspill_pytorch_allocator_failure = _Function()
    shadowspill_pytorch_recover_no_progress = _Function()
    shadowspill_pytorch_profiler_annotations_set = _Function()
    shadowspill_pytorch_allocation_for_pointer = _Function()
    shadowspill_pytorch_allocation_scope_begin = _Function()
    shadowspill_pytorch_allocation_scope_end = _Function()
    shadowspill_pytorch_allocation_scope_abort = _Function()
    shadowspill_pytorch_acquire_objects_handle = _Function()
    shadowspill_pytorch_transfer_acquired_object_to_caller = _Function()
    shadowspill_pytorch_submit_action_batch_handle = _Function()
    shadowspill_pytorch_before_task_handle = _Function()
    shadowspill_pytorch_after_task_handle = _Function()
    shadowspill_pytorch_validate_object_binding = _Function()
    shadowspill_pytorch_abort_task_handle = _Function()


def test_declarative_adapter_abi_has_expected_c_layout() -> None:
    assert ctypes.sizeof(AdapterConfig) == 80
    assert ctypes.sizeof(AdapterCapabilities) == 20
    assert ctypes.sizeof(RuntimeStatistics) == 52 * 8
    assert ctypes.sizeof(AllocationEvent) == 80
    assert ctypes.sizeof(Allocation) == 48
    assert ctypes.sizeof(BackendStatistics) == 22 * 8
    assert ctypes.sizeof(RuntimeFailure) == 192
    assert ctypes.sizeof(AdapterFailure) == 216
    assert ctypes.sizeof(AdapterStatistics) == 672
    assert ctypes.sizeof(ObjectBinding) == 40
    assert ctypes.sizeof(ObjectDescription) == 32
    assert ctypes.sizeof(ObjectUpdate) == 16
    assert ctypes.sizeof(RuntimeAction) == 24
    assert ctypes.sizeof(FixedPlacementDescription) == 56
    assert ctypes.sizeof(FixedDependencyDescription) == 40
    assert ctypes.sizeof(FixedLayoutDescription) == 48
    assert ctypes.sizeof(ObjectSnapshot) == 96
    assert ctypes.sizeof(ObjectLocationSnapshot) == 64
    assert ctypes.sizeof(PhysicalAdmission) == 72
    assert ctypes.sizeof(PhysicalMemory) == 24
    assert ctypes.sizeof(TaskDispatchTiming) == 88
    assert ctypes.sizeof(TraceConfig) == 24
    assert ctypes.sizeof(TraceEvent) == 96
    assert ctypes.sizeof(TraceSummary) == 72
    assert ctypes.sizeof(TransferCalibrationConfig) == 40
    assert ctypes.sizeof(TransferProfile) == 112


def test_adapter_signatures_are_configured_together() -> None:
    library = _Library()
    configure_adapter_library(library)
    assert library.shadowspill_pytorch_adapter_capabilities.restype is ctypes.c_uint32
    assert library.shadowspill_pytorch_profile_range_begin.argtypes == [ctypes.c_char_p]
    assert library.shadowspill_pytorch_profile_range_end.argtypes == [ctypes.c_uint64]
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
    assert library.shadowspill_pytorch_allocator_close.argtypes == []
    assert library.shadowspill_pytorch_allocator_statistics.argtypes == [
        ctypes.POINTER(AdapterStatistics)
    ]
    assert library.shadowspill_pytorch_allocator_failure.argtypes == [
        ctypes.POINTER(AdapterFailure)
    ]
    assert library.shadowspill_pytorch_allocation_for_pointer.argtypes == [
        ctypes.c_uint64,
        ctypes.POINTER(Allocation),
    ]
    assert library.shadowspill_pytorch_validate_object_binding.argtypes == [
        ctypes.c_uint32,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
    ]
    assert library.shadowspill_pytorch_transfer_acquired_object_to_caller.argtypes == [
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.c_size_t,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.POINTER(Allocation),
    ]
    assert library.shadowspill_pytorch_allocation_scope_begin.argtypes == [
        ctypes.c_uint64
    ]
    assert library.shadowspill_pytorch_allocation_scope_end.argtypes == [
        ctypes.c_uint64,
        ctypes.c_size_t,
    ]
    assert library.shadowspill_pytorch_allocation_scope_abort.argtypes == []
    assert library.shadowspill_pytorch_abort_task_handle.argtypes == [
        ctypes.c_size_t,
    ]


class _RuntimeLibrary:
    """Stands in for the neutral runtime the bridge calls plan admission on."""

    shadowspill_plan_bind_object = _Function()
    shadowspill_plan_admit_task = _Function()
    shadowspill_plan_publish_initial_allocation = _Function()
    shadowspill_plan_admit_fixed_layout = _Function()
    shadowspill_plan_admit_object_acquisition = _Function()
    shadowspill_plan_admit_action_batch = _Function()
    shadowspill_object_handle_release = _Function()
    shadowspill_object_release_generation = _Function()
    shadowspill_trace_prepare = _Function()
    shadowspill_trace_begin = _Function()
    shadowspill_trace_end = _Function()
    shadowspill_trace_read = _Function()
    shadowspill_allocation_telemetry_start = _Function()
    shadowspill_allocation_telemetry_stop = _Function()
    shadowspill_allocation_telemetry_read = _Function()
    shadowspill_unregister_object = _Function()
    shadowspill_rekey_object = _Function()
    shadowspill_object_snapshot = _Function()
    shadowspill_object_location_snapshot = _Function()
    shadowspill_read_object = _Function()
    shadowspill_write_object = _Function()
    shadowspill_plan_create = _Function()
    shadowspill_object_handle_acquire = _Function()
    shadowspill_task_publish_allocation = _Function()
    shadowspill_plan_close = _Function()
    shadowspill_plan_destroy = _Function()
    shadowspill_plan_wait_idle = _Function()
    shadowspill_plan_clear_tasks = _Function()
    shadowspill_plan_seal_fixed_layout = _Function()
    shadowspill_runtime_wait_idle = _Function()
    shadowspill_runtime_calibrate_transfer_capabilities = _Function()
    shadowspill_runtime_transfer_profiles = _Function()
    shadowspill_register_object = _Function()


def test_runtime_signatures_are_configured_together() -> None:
    """Plan admission is declared on the neutral runtime, not the adapter.

    The adapter used to wrap each of these to marshal a handle it did not
    own. Declaring them here is what lets the bridge skip that hop.
    """

    library = _RuntimeLibrary()
    configure_runtime_library(library)

    assert library.shadowspill_plan_wait_idle.argtypes == [ctypes.c_size_t]
    assert library.shadowspill_plan_wait_idle.restype == ctypes.c_uint32
    assert library.shadowspill_runtime_wait_idle.argtypes == [ctypes.c_size_t]
    assert library.shadowspill_register_object.argtypes == [
        ctypes.c_size_t,
        ctypes.POINTER(ObjectDescription),
    ]
    assert library.shadowspill_plan_destroy.restype is None
    assert library.shadowspill_plan_bind_object.argtypes == [
        ctypes.c_size_t,
        ctypes.c_uint64,
        ctypes.c_size_t,
        ctypes.c_uint8,
    ]
    assert library.shadowspill_object_handle_release.argtypes == [ctypes.c_size_t]
    assert library.shadowspill_plan_publish_initial_allocation.argtypes == [
        ctypes.c_size_t,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.POINTER(ObjectBinding),
    ]


def test_execution_reservation_accepts_fragmented_dynamic_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(failures_module, "runtime_library", _IdleRuntime)
    class _StatisticsLibrary:
        allocated = 16
        free = 112
        free_prefix = 112
        largest = 112

        def shadowspill_pytorch_allocator_statistics(self, output: object) -> int:
            statistics = ctypes.cast(output, ctypes.POINTER(AdapterStatistics))[0]
            statistics.runtime.allocated_bytes = self.allocated
            statistics.runtime.free_bytes = self.free
            statistics.runtime.free_prefix_bytes = self.free_prefix
            statistics.runtime.largest_free_range_bytes = self.largest
            return 0

    admission = PhysicalAdmission()
    admission.allocator_pool_bytes = 128
    library = _StatisticsLibrary()
    installed = InstalledAllocator(
        library=library,
        allocator=object(),
        path=Path("/adapter"),
        admission=admission,
        fixed_execution_bytes=16,
    )

    assert validate_dynamic_execution_reservation(installed, reserved_bytes=16) == 16
    with pytest.raises(ValueError, match="smaller"):
        validate_dynamic_execution_reservation(installed, reserved_bytes=15)
    # Dynamic admission may consume all compatible ranges.  Persistent state
    # can split the free capacity without requiring one range as large as the
    # complete planning capacity.
    library.allocated = 20
    library.free = 108
    library.free_prefix = 96
    library.largest = 96
    assert validate_dynamic_execution_reservation(installed, reserved_bytes=32) == 20
    with pytest.raises(AllocatorInstallError, match="exceed"):
        validate_dynamic_execution_reservation(installed, reserved_bytes=16)
    library.largest = 64
    assert validate_dynamic_execution_reservation(installed, reserved_bytes=32) == 20
    library.free = 100
    with pytest.raises(AllocatorInstallError, match="accounting"):
        validate_dynamic_execution_reservation(installed, reserved_bytes=32)


def test_missing_callback_symbol_has_field_specific_error() -> None:
    with pytest.raises(AllocatorInstallError, match="missing_callback"):
        _function_pointer(object(), "missing_callback")


def test_installer_rejects_missing_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "shadowspill.pytorch.runtime_adapter.allocator._installed", None
    )
    missing = tmp_path / "libshadowspill_pytorch.so"
    with pytest.raises(AllocatorInstallError, match="does not exist"):
        install_allocator(
            missing,
            device_ordinal=0,
            device_budget_bytes=1,
            provider_headroom_bytes=0,
            **_two_pool_topology(),
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"device_ordinal": -1}, "ordinal"),
        ({"device_budget_bytes": 0}, "budget"),
        ({"provider_headroom_bytes": -1}, "headroom"),
        ({"provider_headroom_bytes": 1024}, "headroom"),
        (
            {
                "pools": (
                    PoolBootstrap(0, 0, 0),
                    PoolBootstrap(1, 1, -1),
                )
            },
            "capacities",
        ),
        ({"worker_poll_nanoseconds": -1}, "poll"),
    ],
)
def test_installer_rejects_invalid_physical_configuration(
    tmp_path: Path, overrides: dict[str, object], message: str
) -> None:
    arguments = {
        "device_ordinal": 0,
        "device_budget_bytes": 1024,
        "provider_headroom_bytes": 0,
        "worker_poll_nanoseconds": 0,
        **_two_pool_topology(),
    }
    arguments.update(overrides)
    with pytest.raises(AllocatorInstallError, match=message):
        install_allocator(tmp_path / "missing.so", **arguments)
