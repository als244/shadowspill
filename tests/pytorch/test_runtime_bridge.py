from __future__ import annotations

import ctypes
from dataclasses import replace
from typing import Any

import torch

from shadowspill.ir import AliasGroupSpec, ObjectRole, ObjectSpec
from shadowspill.pytorch.profiling import (
    TaskAllocationABI,
    TaskAllocationEvent,
    TaskAllocationOperation,
)
from shadowspill.pytorch.runtime_adapter.bridge import (
    RuntimeBridge,
    TaskMemoryEnvelope,
)
from tests.ir._examples import representative_program


class _AbortLibrary:
    def shadowspill_pytorch_abort_task_range(self) -> None:
        self.aborted = True

def test_abort_task_only_closes_the_native_scope() -> None:
    library = _AbortLibrary()
    bridge = RuntimeBridge(library, representative_program())

    bridge.abort_task()

    assert library.aborted


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
