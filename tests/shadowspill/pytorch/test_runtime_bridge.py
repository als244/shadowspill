from __future__ import annotations

import ctypes
from dataclasses import replace
from typing import Any

import torch

from shadowspill.ir import AliasGroupSpec, ObjectRole, ObjectSpec
from shadowspill.pytorch.profiling import (
    TaskAllocationContract,
    TaskAllocationEvent,
    TaskAllocationOperation,
)
from shadowspill.pytorch.runtime_adapter.bridge import (
    RuntimeBridge,
    TaskMemoryEnvelope,
)
from tests.shadowspill.ir._examples import representative_program


class _AbortLibrary:
    def shadowspill_pytorch_abort_task_handle(
        self, task_handle: int, task_id: int
    ) -> int:
        self.aborted = (task_handle, task_id)
        return 0


class _Installed:
    def __init__(self, library: object) -> None:
        self.library = library


class _Runtime:
    def __init__(self, library: object) -> None:
        self._installed = _Installed(library)
        self._next_object_id = 10_000

    def _reserve_persistent_object_ids(
        self, count: int, *, allow_in_progress_plan: bool = False
    ) -> tuple[int, ...]:
        del allow_in_progress_plan
        first = self._next_object_id
        self._next_object_id += count
        return tuple(range(first, first + count))

    def _reserve_runtime_object_ids(self, count: int) -> tuple[int, ...]:
        return self._reserve_persistent_object_ids(count)


def test_abort_task_only_closes_the_native_scope() -> None:
    library = _AbortLibrary()
    bridge = RuntimeBridge(_Runtime(library), representative_program(), 1)  # type: ignore[arg-type]

    bridge.abort_task(37, 11)

    assert library.aborted == (37, 11)


def test_execution_buffers_project_pointer_free_allocation_contract() -> None:
    program = representative_program()
    bridge = RuntimeBridge(_Runtime(object()), program, 1)  # type: ignore[arg-type]
    trace = (
        TaskAllocationEvent(0, TaskAllocationOperation.ALLOCATE, 64, 64),
        TaskAllocationEvent(0, TaskAllocationOperation.FREE, 64, 64),
    )
    allocation_contract = TaskAllocationContract.capture(trace)

    buffers = bridge._execution_buffers(
        replace(program.tasks[0], task_id="task_000000"),
        (),
        (),
        (),
        TaskMemoryEnvelope(allocation_contract=allocation_contract),
    )

    assert buffers.description.enforce_allocation_contract == 1
    assert buffers.description.allocation_contract_step_count == 2
    assert buffers.allocation_contract_steps[0].operation == 0
    assert buffers.allocation_contract_steps[0].allocation_ordinal == 0
    assert buffers.allocation_contract_steps[0].requested_bytes == 64
    assert buffers.allocation_contract_steps[1].operation == 1


class _LabelLibrary:
    def __init__(self) -> None:
        self.labels: tuple[str, ...] = ()

    def shadowspill_pytorch_task_labels_configure(self, values: Any, count: int) -> int:
        labels = ctypes.cast(values, ctypes.POINTER(ctypes.c_char_p))
        self.labels = tuple(labels[index].decode() for index in range(count))
        return 0


def test_task_trace_labels_are_projected_by_plan_local_id() -> None:
    library = _LabelLibrary()
    bridge = RuntimeBridge(_Runtime(library), representative_program(), 1)  # type: ignore[arg-type]

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
    bridge = RuntimeBridge(_Runtime(object()), program, 1)  # type: ignore[arg-type]
    tensor = torch.empty(0)

    bridge.register_placeholder("alias_000099")
    binding = bridge.bind_registered_tensor("alias_000099", tensor)
    bridge.rebind_many(((tensor, "alias_000099", binding),))

    assert not bridge.requires_storage("alias_000099")
    assert binding.pointer is None
    assert binding.generation == 0
