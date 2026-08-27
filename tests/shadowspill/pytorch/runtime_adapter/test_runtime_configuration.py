from __future__ import annotations

import ctypes
import threading
from types import SimpleNamespace

import pytest

from shadowspill.errors import (
    AdmissionError,
)
from shadowspill.pytorch.runtime_adapter import runtime as runtime_module
from shadowspill.pytorch.runtime_adapter.runtime import (
    MemoryPool,
    Runtime,
    _resolve_dynamic_scratch_reserve,
    _resolve_execution_budget,
)


def test_execution_budget_accepts_runtime_physical_cap_without_double_charge() -> None:
    physical = 16 << 30
    derived = physical - (1280 << 20) - (256 << 20)
    pool = MemoryPool("execution", 0, "device", derived, physical, 0)

    assert _resolve_execution_budget(None, pool) == derived
    assert _resolve_execution_budget(physical, pool) == derived
    assert _resolve_execution_budget(10 << 30, pool) == 10 << 30
    with pytest.raises(AdmissionError, match="falls between"):
        _resolve_execution_budget(derived + 1, pool)
    with pytest.raises(AdmissionError, match="physical capacity"):
        _resolve_execution_budget(physical + 1, pool)


def test_dynamic_scratch_reserve_is_an_optional_bounded_minimum() -> None:
    budget = 16 << 30
    assert _resolve_dynamic_scratch_reserve(None, execution_budget=budget) == 0
    assert _resolve_dynamic_scratch_reserve(1 << 30, execution_budget=budget) == (
        1 << 30
    )
    with pytest.raises(TypeError, match="integer byte count"):
        _resolve_dynamic_scratch_reserve(True, execution_budget=budget)
    with pytest.raises(AdmissionError, match="non-negative"):
        _resolve_dynamic_scratch_reserve(-1, execution_budget=budget)
    with pytest.raises(AdmissionError, match="exceeds"):
        _resolve_dynamic_scratch_reserve(budget + 1, execution_budget=budget)


def test_runtime_object_reference_owns_and_releases_one_runtime_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acquiring needs the adapter's bound runtime; releasing needs only the
    handle, so the two halves are answered by different libraries."""

    class _RuntimeLibrary:
        def __init__(self) -> None:
            self.released: list[int] = []

        def shadowspill_object_handle_acquire(
            self, runtime_handle: int, object_id: int, output: object
        ) -> int:
            assert object_id == 41
            ctypes.cast(output, ctypes.POINTER(ctypes.c_size_t))[0] = 73
            return 0

        def shadowspill_object_handle_release(self, handle: int) -> int:
            self.released.append(handle)
            return 0

    neutral = _RuntimeLibrary()
    monkeypatch.setattr(runtime_module, "runtime_library", lambda: neutral)
    runtime = Runtime.__new__(Runtime)
    runtime._lock = threading.RLock()
    runtime._closed = False
    runtime._unusable_reason = None
    runtime._installed = SimpleNamespace(library=object())
    runtime._runtime_handle = 0
    runtime._active_object_references = 0

    reference = runtime._acquire_object_reference(object_id=41, size_bytes=2048)

    assert reference.object_id == 41
    assert reference.size_bytes == 2048
    assert runtime._active_object_references == 1
    reference.close()
    reference.close()
    assert neutral.released == [73]
    assert runtime._active_object_references == 0
