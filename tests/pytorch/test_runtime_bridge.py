from __future__ import annotations

import ctypes

import pytest

from shadowspill.pytorch.runtime_bridge import RuntimeBridge, RuntimeExecutionError
from tests.ir._examples import representative_program


class _FailureLibrary:
    def shadowspill_pytorch_abort_task_range(self) -> None:
        self.aborted = True

    def shadowspill_pytorch_allocator_failure(self, output: object) -> int:
        failure = ctypes.cast(output, ctypes.POINTER(ctypes.c_uint8))
        del failure
        from shadowspill.pytorch._abi import AdapterFailure

        result = ctypes.cast(output, ctypes.POINTER(AdapterFailure))[0]
        result.status = 4
        result.device_ordinal = 0
        result.requested_bytes = 525_336_576
        result.runtime.status = 4
        result.runtime.object_id = (1 << 64) - 1
        result.runtime.allocation_id = (1 << 64) - 1
        result.runtime.requested_bytes = 525_336_576
        result.runtime.free_bytes = 600_000_000
        result.runtime.largest_free_range_bytes = 400_000_000
        return 4


def test_failed_task_translates_latched_allocator_diagnostics() -> None:
    library = _FailureLibrary()
    bridge = RuntimeBridge(library, representative_program())
    original = RuntimeError("tensor data is not allocated")

    with pytest.raises(RuntimeExecutionError) as caught:
        bridge.abort_task_after_failure("execute task task_000007", original)

    assert library.aborted
    assert caught.value.__cause__ is original
    message = str(caught.value)
    assert "task_000007" in message
    assert "requested=525336576" in message
    assert "free=600000000" in message
    assert "largest_free_range=400000000" in message


class _HealthyLibrary:
    def shadowspill_pytorch_abort_task_range(self) -> None:
        self.aborted = True

    def shadowspill_pytorch_allocator_failure(self, output: object) -> int:
        del output
        return 0


def test_failed_task_preserves_original_error_without_allocator_failure() -> None:
    library = _HealthyLibrary()
    bridge = RuntimeBridge(library, representative_program())
    original = RuntimeError("ordinary operation failure")

    bridge.abort_task_after_failure("execute task task_000007", original)

    assert library.aborted
