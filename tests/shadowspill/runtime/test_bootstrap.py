from __future__ import annotations

import ctypes
from pathlib import Path

import pytest

from shadowspill.runtime import failures as failures_module
from shadowspill.runtime.abi import (
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
    MemoryPoolStatistics,
    ObjectBinding,
    ObjectDescription,
    ObjectLocationSnapshot,
    ObjectSnapshot,
    ObjectUpdate,
    PhysicalAdmission,
    PhysicalMemory,
    PoolConfig,
    RuntimeAction,
    RuntimeFailure,
    RuntimeStatistics,
    TraceConfig,
    TraceEvent,
    TraceSummary,
    TransferCalibrationConfig,
    TransferProfile,
    configure_adapter_library,
    configure_runtime_library,
)
from shadowspill.runtime.abi.signatures import _RUNTIME_SIGNATURES
from shadowspill.runtime.bootstrap import (
    InstalledRuntime,
    PoolBootstrap,
    RouteBootstrap,
    RuntimeInstallError,
    _function_pointer,
    install_runtime,
    validate_dynamic_execution_reservation,
)


class _NoProcessAllocator:
    """A frontend whose allocator is never reached: every case here refuses
    before installation, or exercises the accounting after it."""

    def refuse_unusable_build(self) -> None:
        raise AssertionError("the request should have been refused first")

    def missing_operations(self) -> tuple[str, ...]:
        return ()

    def prepare(self, library_path: Path, record_stream_pointer: int) -> None:
        raise AssertionError("the request should have been refused first")

    def activate(self) -> None:
        raise AssertionError("the request should have been refused first")

    def initialize_provider_workspaces(self, device_ordinal: int) -> None:
        return None


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
    shadowspill_pytorch_physical_admission = _Function()
    shadowspill_pytorch_physical_memory = _Function()
    shadowspill_pytorch_seal_physical_budget = _Function()
    shadowspill_pytorch_check_physical_budget = _Function()
    shadowspill_pytorch_allocator_bootstrap = _Function()
    shadowspill_pytorch_allocator_close = _Function()
    shadowspill_pytorch_allocator_statistics = _Function()
    shadowspill_pytorch_allocator_failure = _Function()
    shadowspill_pytorch_recover_no_progress = _Function()
    shadowspill_pytorch_allocation_for_pointer = _Function()
    shadowspill_pytorch_allocation_scope_begin = _Function()
    shadowspill_pytorch_allocation_scope_end = _Function()
    shadowspill_pytorch_allocation_scope_abort = _Function()
    shadowspill_pytorch_transfer_acquired_object_to_caller = _Function()
    shadowspill_pytorch_before_task_handle = _Function()
    shadowspill_pytorch_after_task_handle = _Function()
    shadowspill_pytorch_validate_object_binding = _Function()
    shadowspill_pytorch_abort_task_handle = _Function()


def test_declarative_adapter_abi_has_expected_c_layout() -> None:
    # Two pointers and two counts past the original 88: the extension library
    # list, and a per-pool configuration forwarded to its kind.
    assert ctypes.sizeof(AdapterConfig) == 104
    assert ctypes.sizeof(PoolConfig) == 24
    assert ctypes.sizeof(AdapterCapabilities) == 16
    # A uint32 pool count, padded, then the runtime-wide counters. A pool's own
    # numbers are in MemoryPoolStatistics: a uint32 id and a uint8 kind, padded,
    # then its counters.
    assert ctypes.sizeof(RuntimeStatistics) == 8 + 30 * 8
    assert ctypes.sizeof(MemoryPoolStatistics) == 8 + 19 * 8
    assert ctypes.sizeof(AllocationEvent) == 80
    assert ctypes.sizeof(Allocation) == 48
    # 23 since `write_value` joined the contract and the mock counts its
    # stream writes beside its stream waits.
    assert ctypes.sizeof(BackendStatistics) == 23 * 8
    assert ctypes.sizeof(RuntimeFailure) == 192
    assert ctypes.sizeof(AdapterFailure) == 216
    # 672 rather than 664: it embeds BackendStatistics, which grew by one.
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
    assert ctypes.sizeof(TraceConfig) == 24
    # Each grew by one uint64 when a transfer gained a third instant: the event
    # carries `lane_issued_at_ns` beside the two it had, and the summary carries
    # `origin_host_ns`, the anchor a lane off the device clock converts through.
    assert ctypes.sizeof(TraceEvent) == 104
    assert ctypes.sizeof(TraceSummary) == 80
    assert ctypes.sizeof(TransferCalibrationConfig) == 40
    assert ctypes.sizeof(TransferProfile) == 112


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
        ctypes.c_uint64,
        ctypes.c_uint64,
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
    """Stands in for the neutral runtime the bridge calls plan admission on.

    Every name the signature table declares is present, so a signature added
    to the runtime never has to be added here as well; what the test checks is
    the argument and result types each one is given.
    """

    def __init__(self) -> None:
        for name, _arguments, _result in _RUNTIME_SIGNATURES:
            setattr(self, name, _Function())


def test_runtime_signatures_are_configured_together() -> None:
    """Plan admission is declared on the neutral runtime, not the adapter.

    The adapter used to wrap each of these to marshal a handle it did not
    own. Declaring them here is what lets the bridge skip that hop.
    """

    library = _RuntimeLibrary()
    configure_runtime_library(library)

    # Profiling is the runtime's: the adapter keeps no profiler of its own, so
    # these are declared here and take the runtime handle.
    assert library.shadowspill_profiler_range_begin.argtypes == [
        ctypes.c_size_t,
        ctypes.c_char_p,
    ]
    assert library.shadowspill_profiler_range_end.argtypes == [
        ctypes.c_size_t,
        ctypes.c_uint64,
    ]
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
            statistics.allocator_pool.allocated_bytes = self.allocated
            statistics.allocator_pool.free_bytes = self.free
            statistics.allocator_pool.free_prefix_bytes = self.free_prefix
            statistics.allocator_pool.largest_free_range_bytes = self.largest
            return 0

    admission = PhysicalAdmission()
    admission.allocator_pool_bytes = 128
    library = _StatisticsLibrary()
    installed = InstalledRuntime(
        library=library,
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
    with pytest.raises(RuntimeInstallError, match="exceed"):
        validate_dynamic_execution_reservation(installed, reserved_bytes=16)
    library.largest = 64
    assert validate_dynamic_execution_reservation(installed, reserved_bytes=32) == 20
    library.free = 100
    with pytest.raises(RuntimeInstallError, match="accounting"):
        validate_dynamic_execution_reservation(installed, reserved_bytes=32)


def test_missing_callback_symbol_has_field_specific_error() -> None:
    with pytest.raises(RuntimeInstallError, match="missing_callback"):
        _function_pointer(object(), "missing_callback")


def test_installer_rejects_missing_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("shadowspill.runtime.bootstrap._installed", None)
    missing = tmp_path / "libshadowspill_pytorch.so"
    with pytest.raises(RuntimeInstallError, match="does not exist"):
        install_runtime(
            missing,
            frontend=_NoProcessAllocator(),
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
    with pytest.raises(RuntimeInstallError, match=message):
        install_runtime(
            tmp_path / "missing.so",
            frontend=_NoProcessAllocator(),
            **arguments,
        )
