"""The bridge: task encoding, the abort boundary, zero-byte aliases."""

from __future__ import annotations

import threading
from dataclasses import replace

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
    abort_task,
    encode_task,
    publish_initial_tensor,
    rebind_many,
)
from tests.shadowspill.ir._examples import representative_program


class _AbortLibrary:
    def shadowspill_pytorch_abort_task_handle(self, task_handle: int) -> int:
        self.aborted = task_handle
        return 0


class _Installed:
    def __init__(self, library: object) -> None:
        self.library = library


class _Runtime:
    """What `objects.reserve_runtime_object_ids` reads of a runtime."""

    def __init__(self, library: object) -> None:
        self._installed = _Installed(library)
        self._lock = threading.RLock()
        self._next_persistent_object_id = 10_000

    def _require_open(self) -> None:
        return None


def test_abort_task_only_closes_the_runtime_scope() -> None:
    library = _AbortLibrary()
    bridge = RuntimeBridge(  # type: ignore[arg-type]
        _Runtime(library),
        representative_program(),
        1,
        execution_pool_id=0,
        spill_pool_id=1,
    )

    abort_task(bridge, 37)

    assert library.aborted == 37


def test_execution_buffers_project_pointer_free_allocation_contract() -> None:
    program = representative_program()
    bridge = RuntimeBridge(  # type: ignore[arg-type]
        _Runtime(object()),
        program,
        1,
        execution_pool_id=0,
        spill_pool_id=1,
    )
    trace = (
        TaskAllocationEvent(0, TaskAllocationOperation.ALLOCATE, 64, 64),
        TaskAllocationEvent(0, TaskAllocationOperation.FREE, 64, 64),
    )
    allocation_contract = TaskAllocationContract.capture(trace)

    buffers = encode_task(
        bridge,
        replace(program.tasks[0], task_id="task_000000"),
        (),
        (),
        (),
        (),
        TaskMemoryEnvelope(allocation_contract=allocation_contract),
        "execution_000000.forward.stage_0000",
    )

    assert buffers.description.enforce_allocation_contract == 1
    assert buffers.description.allocation_contract_step_count == 2
    assert buffers.allocation_contract_steps[0].operation == 0
    assert buffers.allocation_contract_steps[0].allocation_ordinal == 0
    assert buffers.allocation_contract_steps[0].requested_bytes == 64
    assert buffers.allocation_contract_steps[1].operation == 1


def test_task_trace_label_is_owned_by_the_admitted_description() -> None:
    bridge = RuntimeBridge(  # type: ignore[arg-type]
        _Runtime(object()),
        representative_program(),
        1,
        execution_pool_id=0,
        spill_pool_id=1,
    )
    buffers = encode_task(
        bridge,
        replace(representative_program().tasks[0], task_id="task_000000"),
        (),
        (),
        (),
        (),
        TaskMemoryEnvelope(),
        "execution_000000.forward.stage_0000",
    )

    assert buffers.description.trace_label == (b"execution_000000.forward.stage_0000")


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
    bridge = RuntimeBridge(  # type: ignore[arg-type]
        _Runtime(object()),
        program,
        1,
        execution_pool_id=0,
        spill_pool_id=1,
    )
    tensor = torch.empty(0)

    bridge.objects.register_placeholder("alias_000099")
    binding = publish_initial_tensor(bridge, "alias_000099", tensor)
    rebind_many(bridge, ((tensor, "alias_000099", binding),))

    assert not bridge.objects.requires_storage("alias_000099")
    assert binding.pointer is None
    assert binding.generation == 0
