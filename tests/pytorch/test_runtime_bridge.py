from __future__ import annotations

import ctypes
from dataclasses import replace
from typing import Any

import pytest
import torch

from shadowspill.ir import AliasGroupSpec, ObjectRole, ObjectSpec
from shadowspill.pytorch.runtime_adapter.bridge import (
    RuntimeBridge,
    RuntimeExecutionError,
)
from shadowspill.pytorch.runtime_adapter.failures import ExecutionTaskIdentity
from tests.ir._examples import representative_program


class _FailureLibrary:
    def shadowspill_pytorch_abort_task_range(self) -> None:
        self.aborted = True

    def shadowspill_pytorch_allocator_failure(self, output: Any) -> int:
        from shadowspill.pytorch.runtime_adapter.abi import AdapterFailure

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
    task = ExecutionTaskIdentity(
        "execution_000017",
        "microbatch_0001.stage_0004.backward.recompute",
        "task_000007",
    )

    with pytest.raises(RuntimeExecutionError) as caught:
        bridge.abort_task_after_failure("execute task task_000007", original, task=task)

    assert library.aborted
    assert caught.value.__cause__ is original
    message = str(caught.value)
    assert "ShadowSpill no-progress OOM" in message
    assert "execution_task: execution_000017" in message
    assert "semantic_task: microbatch_0001.stage_0004.backward.recompute" in message
    assert "canonical_task: task_000007" in message
    assert "requested: 525336576" in message
    assert "free: 600000000" in message
    assert "largest_free_range: 400000000" in message
    assert caught.value.diagnostics is not None
    assert caught.value.diagnostics.as_dict()["execution_task_id"] == (
        "execution_000017"
    )


class _HealthyLibrary:
    def shadowspill_pytorch_abort_task_range(self) -> None:
        self.aborted = True

    def shadowspill_pytorch_allocator_failure(self, output: Any) -> int:
        del output
        return 0


def test_failed_task_preserves_original_error_without_allocator_failure() -> None:
    library = _HealthyLibrary()
    bridge = RuntimeBridge(library, representative_program())
    original = RuntimeError("ordinary operation failure")

    bridge.abort_task_after_failure("execute task task_000007", original)

    assert library.aborted


class _BackendFailureLibrary(_FailureLibrary):
    def shadowspill_pytorch_allocator_failure(self, output: Any) -> int:
        from shadowspill.pytorch.runtime_adapter.abi import AdapterFailure

        result = ctypes.cast(output, ctypes.POINTER(AdapterFailure))[0]
        result.status = 7
        result.device_ordinal = 0
        result.runtime.status = 7
        return 7


def test_failed_task_preserves_original_error_for_non_oom_runtime_failure() -> None:
    library = _BackendFailureLibrary()
    bridge = RuntimeBridge(library, representative_program())
    original = RuntimeError("CUDA illegal memory access")

    bridge.abort_task_after_failure("execute task task_000007", original)

    assert library.aborted


class _PlacementFailureLibrary(_FailureLibrary):
    def shadowspill_pytorch_allocator_failure(self, output: Any) -> int:
        from shadowspill.pytorch.runtime_adapter.abi import AdapterFailure

        result = ctypes.cast(output, ctypes.POINTER(AdapterFailure))[0]
        result.status = 6
        result.device_ordinal = 0
        result.runtime.status = 6
        result.runtime.task_id = 28
        result.runtime.object_id = (1 << 64) - 1
        result.runtime.allocation_id = (1 << 64) - 1
        result.runtime.requested_bytes = 2_097_152
        result.runtime.free_bytes = 3_603_365_636
        result.runtime.largest_free_range_bytes = 2_553_282_560
        result.runtime.allocation_ordinal = 9
        result.runtime.expected_allocation_ordinal = 9
        result.runtime.expected_requested_bytes = 16_384
        return 6


def test_failed_task_surfaces_allocation_placement_contract() -> None:
    library = _PlacementFailureLibrary()
    bridge = RuntimeBridge(library, representative_program())
    original = RuntimeError("tensor data is not allocated")
    task = ExecutionTaskIdentity(
        "execution_000028",
        "microbatch_0000.stage_0028.forward.recompute",
        "task_000028",
    )

    with pytest.raises(RuntimeExecutionError) as caught:
        bridge.abort_task_after_failure("execute task task_000028", original, task=task)

    assert library.aborted
    assert caught.value.__cause__ is original
    message = str(caught.value)
    assert "status 6 (plan_violation)" in message
    assert "execution_task: execution_000028" in message
    assert "allocation_ordinal: 9" in message
    assert "expected_allocation_ordinal: 9" in message
    assert "expected_requested: 16384" in message


class _LabelLibrary:
    def __init__(self) -> None:
        self.labels: tuple[str, ...] = ()

    def shadowspill_pytorch_task_labels_configure(
        self, values: Any, count: int
    ) -> int:
        labels = ctypes.cast(values, ctypes.POINTER(ctypes.c_char_p))
        self.labels = tuple(labels[index].decode() for index in range(count))
        return 0


def test_task_trace_labels_are_projected_by_dense_canonical_id() -> None:
    library = _LabelLibrary()
    bridge = RuntimeBridge(library, representative_program())

    bridge.configure_task_labels(
        {
            "task_000002": "execution_000001.backward.stage_0001",
            "task_000000": "execution_000000.forward.stage_0000",
        }
    )

    assert library.labels == (
        "execution_000000.forward.stage_0000",
        "",
        "execution_000001.backward.stage_0001",
    )


def test_zero_size_alias_uses_no_physical_runtime_operation() -> None:
    program = representative_program()
    program = replace(
        program,
        alias_groups=(
            *program.alias_groups,
            AliasGroupSpec("alias_000099", "cuda_0", 0),
        ),
        objects=(
            *program.objects,
            ObjectSpec(
                "object_000099",
                "alias_000099",
                0,
                0,
                ObjectRole.ACTIVATION,
            ),
        ),
    )
    bridge = RuntimeBridge(object(), program)
    tensor = torch.empty(0)

    bridge.register_placeholder("alias_000099")
    binding = bridge.bind_registered_tensor("alias_000099", tensor)
    generations = bridge.adopt_many(((tensor, "alias_000099"),))
    bridge.rebind_many(((tensor, "alias_000099", binding),))

    assert not bridge.requires_storage("alias_000099")
    assert binding.pointer is None
    assert binding.generation == 0
    assert generations == (0,)
