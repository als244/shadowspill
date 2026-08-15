from __future__ import annotations

import ctypes
from dataclasses import replace
from typing import Any

import pytest
import torch

from shadowspill.ir import AliasGroupSpec, ObjectRole, ObjectSpec
from shadowspill.pytorch.profiling import (
    TaskAllocationABI,
    TaskAllocationEvent,
    TaskAllocationOperation,
)
from shadowspill.pytorch.runtime_adapter.bridge import (
    RuntimeBridge,
    RuntimeExecutionError,
    TaskMemoryEnvelope,
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


class _EnvelopeFailureLibrary(_FailureLibrary):
    def shadowspill_pytorch_allocator_failure(self, output: Any) -> int:
        from shadowspill.pytorch.runtime_adapter.abi import AdapterFailure

        result = ctypes.cast(output, ctypes.POINTER(AdapterFailure))[0]
        result.status = 10
        result.device_ordinal = 0
        result.runtime.status = 10
        result.runtime.task_id = 28
        result.runtime.object_id = (1 << 64) - 1
        result.runtime.allocation_id = (1 << 64) - 1
        result.runtime.requested_bytes = 2_097_152
        result.runtime.free_bytes = 3_603_365_636
        result.runtime.largest_free_range_bytes = 2_553_282_560
        result.runtime.task_live_requested_bytes = 18_874_368
        result.runtime.task_live_charged_bytes = 18_874_368
        result.runtime.task_live_requested_limit_bytes = 16_777_216
        result.runtime.task_live_charged_limit_bytes = 16_777_216
        result.runtime.task_maximum_requested_allocation_bytes = 2_097_152
        result.runtime.task_maximum_charged_allocation_bytes = 2_097_152
        return 10


def test_failed_task_surfaces_allocation_envelope_contract() -> None:
    library = _EnvelopeFailureLibrary()
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
    assert "status 10 (task_allocation_envelope_exceeded)" in message
    assert "execution_task: execution_000028" in message
    assert "task_live_requested: 18874368" in message
    assert "task_live_requested_limit: 16777216" in message
    assert "task_live_charged: 18874368" in message
    assert "task_live_charged_limit: 16777216" in message


class _AllocationABIFailureLibrary(_FailureLibrary):
    def shadowspill_pytorch_allocator_failure(self, output: Any) -> int:
        from shadowspill.pytorch.runtime_adapter.abi import AdapterFailure

        result = ctypes.cast(output, ctypes.POINTER(AdapterFailure))[0]
        result.status = 11
        result.device_ordinal = 0
        result.runtime.status = 11
        result.runtime.task_id = 28
        result.runtime.object_id = (1 << 64) - 1
        result.runtime.allocation_id = (1 << 64) - 1
        result.runtime.task_allocation_operation_index = 7
        result.runtime.task_allocation_expected_ordinal = 4
        result.runtime.task_allocation_actual_ordinal = 4
        result.runtime.task_allocation_expected_requested_bytes = 4096
        result.runtime.task_allocation_actual_requested_bytes = 8192
        result.runtime.task_allocation_expected_charged_bytes = 4096
        result.runtime.task_allocation_actual_charged_bytes = 8192
        result.runtime.task_allocation_expected_alignment_bytes = 256
        result.runtime.task_allocation_actual_alignment_bytes = 256
        result.runtime.task_allocation_expected_operation = 0
        result.runtime.task_allocation_actual_operation = 0
        return 11


def test_failed_task_surfaces_allocation_abi_contract() -> None:
    library = _AllocationABIFailureLibrary()
    bridge = RuntimeBridge(library, representative_program())
    original = RuntimeError("tensor data is not allocated")

    with pytest.raises(RuntimeExecutionError) as caught:
        bridge.abort_task_after_failure("execute task task_000028", original)

    message = str(caught.value)
    assert "status 11 (task_allocation_abi_mismatch)" in message
    assert "reason: TASK_ALLOCATION_ABI_MISMATCH" in message
    assert "task_allocation_operation_index: 7" in message
    assert "expected_operation: allocate" in message
    assert "expected_requested: 4096" in message
    assert "actual_requested: 8192" in message


def test_execution_buffers_project_pointer_free_allocation_abi() -> None:
    program = representative_program()
    bridge = RuntimeBridge(object(), program)
    trace = (
        TaskAllocationEvent(0, TaskAllocationOperation.ALLOCATE, 64, 64),
        TaskAllocationEvent(0, TaskAllocationOperation.FREE, 64, 64),
    )
    allocation_abi = TaskAllocationABI.capture(trace)

    buffers = bridge._execution_buffers(
        replace(program.tasks[0], task_id="task_000000"),
        (),
        (),
        (),
        TaskMemoryEnvelope(allocation_abi=allocation_abi),
    )

    assert buffers.description.enforce_allocation_abi == 1
    assert buffers.description.allocation_abi_step_count == 2
    assert buffers.allocation_abi_steps[0].operation == 0
    assert buffers.allocation_abi_steps[0].allocation_ordinal == 0
    assert buffers.allocation_abi_steps[0].requested_bytes == 64
    assert buffers.allocation_abi_steps[1].operation == 1


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
