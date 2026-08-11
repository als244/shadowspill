from __future__ import annotations

import ctypes
from pathlib import Path

import pytest

from shadowspill.pytorch._abi import (
    AdapterCapabilities,
    AdapterConfig,
    AdapterFailure,
    AdapterStatistics,
    CudaStatistics,
    ObjectBinding,
    ObjectSnapshot,
    ObjectUpdate,
    RuntimeAction,
    RuntimeFailure,
    RuntimeStatistics,
    configure_adapter_library,
)
from shadowspill.pytorch._allocator import (
    AllocatorInstallError,
    _function_pointer,
    install_allocator,
)


class _Function:
    argtypes: object = None
    restype: object = None


class _Library:
    shadowspill_pytorch_adapter_capabilities = _Function()
    shadowspill_pytorch_allocator_bootstrap = _Function()
    shadowspill_pytorch_allocator_statistics = _Function()
    shadowspill_pytorch_allocator_failure = _Function()
    shadowspill_pytorch_allocator_wait_idle = _Function()
    shadowspill_pytorch_promote_allocation = _Function()
    shadowspill_pytorch_before_task = _Function()
    shadowspill_pytorch_after_task = _Function()
    shadowspill_pytorch_object_snapshot = _Function()
    shadowspill_pytorch_abort_task_range = _Function()


def test_declarative_adapter_abi_has_expected_c_layout() -> None:
    assert ctypes.sizeof(AdapterConfig) == 32
    assert ctypes.sizeof(AdapterCapabilities) == 16
    assert ctypes.sizeof(RuntimeStatistics) == 19 * 8
    assert ctypes.sizeof(CudaStatistics) == 16 * 8
    assert ctypes.sizeof(RuntimeFailure) == 48
    assert ctypes.sizeof(AdapterFailure) == 72
    assert ctypes.sizeof(AdapterStatistics) == 328
    assert ctypes.sizeof(ObjectBinding) == 40
    assert ctypes.sizeof(ObjectUpdate) == 16
    assert ctypes.sizeof(RuntimeAction) == 16
    assert ctypes.sizeof(ObjectSnapshot) == 72


def test_adapter_signatures_are_configured_together() -> None:
    library = _Library()
    configure_adapter_library(library)
    assert library.shadowspill_pytorch_adapter_capabilities.restype is ctypes.c_uint32
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


def test_missing_callback_symbol_has_field_specific_error() -> None:
    with pytest.raises(AllocatorInstallError, match="missing_callback"):
        _function_pointer(object(), "missing_callback")


def test_installer_rejects_missing_library(tmp_path: Path) -> None:
    missing = tmp_path / "libshadowspill_pytorch.so"
    with pytest.raises(AllocatorInstallError, match="does not exist"):
        install_allocator(
            missing,
            device_ordinal=0,
            device_slab_bytes=1,
            host_arena_bytes=0,
        )
