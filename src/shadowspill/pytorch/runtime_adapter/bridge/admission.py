"""Plan-time admission: tasks, the fixed layout, action batches, acquisitions.

Before a plan runs, every task it will execute, the fixed physical layout its
objects are placed in, the initial-placement action batches and the public
object sets a caller may acquire are described to the neutral runtime once,
through the plan handle, and answered with the handles the boundaries use.
`encode_task` is the ctypes description of one task and the buffers that keep it
alive until the runtime has copied it.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from shadowspill.ir import MemoryAction, MutationSpec, TaskSpec
from shadowspill.runtime.abi import (
    FixedDependencyDescription,
    FixedLayoutDescription,
    FixedPlacementDescription,
    ObjectUpdate,
    RuntimeAction,
    TaskAllocationContractStep,
    TaskDescription,
    TaskPublicationDescription,
)
from shadowspill.runtime.failures import RuntimeExecutionError
from shadowspill.runtime.fixed_layout import RuntimeFixedLayout
from shadowspill.status import ABI_VERSION

from .common import (
    ACTION_KIND,
    TaskMemoryEnvelope,
    TaskPublication,
    action_labels,
    plan_local_id,
    runtime_action,
)
from .report import describe_pool_occupants, statistics

if TYPE_CHECKING:
    from . import RuntimeBridge


@dataclass(slots=True)
class EncodedTask:
    """One task's ctypes description, with the buffers it points into."""

    description: TaskDescription
    encoded_task_label: bytes
    input_ids: Any
    updates: Any
    publications: Any
    actions: Any
    allocation_contract_steps: Any
    encoded_labels: tuple[bytes | None, ...]


def admit_task(
    bridge: RuntimeBridge,
    task: TaskSpec,
    input_alias_ids: tuple[str, ...],
    actions: tuple[MemoryAction, ...],
    action_trace_labels: tuple[str, ...] | None = None,
    memory_envelope: TaskMemoryEnvelope | None = None,
    *,
    trace_label: str,
    publications: tuple[TaskPublication, ...] = (),
) -> int:
    """Resolve one immutable task topology in the neutral runtime."""

    if memory_envelope is None:
        memory_envelope = TaskMemoryEnvelope()
    labels = action_labels(actions, action_trace_labels)
    runtime_inputs = _runtime_inputs(bridge, input_alias_ids)
    mutations = _runtime_mutations(bridge, task.mutations)
    action_pairs = _runtime_actions(bridge, actions, labels)
    runtime_publications = tuple(
        item for item in publications if bridge.objects.requires_storage(item.alias_id)
    )
    for alias_id in _referenced_aliases(
        bridge,
        runtime_inputs,
        mutations,
        runtime_publications,
        action_pairs,
    ):
        bridge.objects.register_placeholder(alias_id)
    bridge.objects.bind(
        _referenced_aliases(
            bridge, runtime_inputs, mutations, runtime_publications, action_pairs
        )
    )
    buffers = encode_task(
        bridge,
        task,
        runtime_inputs,
        mutations,
        runtime_publications,
        action_pairs,
        memory_envelope,
        trace_label,
    )
    task_handle = ctypes.c_size_t()
    bridge.require(
        bridge.runtime_library.shadowspill_plan_admit_task(
            bridge.plan_handle,
            ctypes.byref(buffers.description),
            ctypes.byref(task_handle),
        ),
        f"admit task {task.task_id}",
    )
    if task_handle.value == 0:
        raise RuntimeExecutionError(f"task {task.task_id} admitted with a null handle")
    resolved = int(task_handle.value)
    bridge._admitted_task_handles.add(resolved)
    return resolved


def admit_fixed_layout(bridge: RuntimeBridge, layout: RuntimeFixedLayout) -> None:
    """Copy one indexed physical-layout certificate into the C runtime."""

    if bridge._admitted_task_handles or bridge._admitted_action_batches:
        raise RuntimeExecutionError(
            "fixed layout must be admitted before execution tasks"
        )
    placements = (FixedPlacementDescription * len(layout.placements))(
        *(
            FixedPlacementDescription(
                task_id=item.task_id,
                ordinal=item.ordinal,
                object_id=item.object_id,
                offset=item.offset,
                bytes=item.bytes,
                alignment_bytes=item.alignment,
                kind=int(item.kind),
            )
            for item in layout.placements
        )
    )
    dependencies = (FixedDependencyDescription * len(layout.dependencies))(
        *(
            FixedDependencyDescription(
                predecessor_task_id=item.predecessor_task_id,
                predecessor_action_ordinal=(item.predecessor_action_ordinal),
                successor_task_id=item.successor_task_id,
                successor_ordinal=item.successor_ordinal,
                successor_kind=int(item.successor_kind),
            )
            for item in layout.dependencies
        )
    )
    description = FixedLayoutDescription(
        abi_version=ABI_VERSION,
        slice_bytes=layout.slice_bytes,
        placements=placements if layout.placements else None,
        placement_count=len(layout.placements),
        dependencies=dependencies if layout.dependencies else None,
        dependency_count=len(layout.dependencies),
    )
    status = int(
        bridge.runtime_library.shadowspill_plan_admit_fixed_layout(
            bridge.plan_handle, ctypes.byref(description)
        )
    )
    if status != 0:
        pool = statistics(bridge).allocator_pool
        raise RuntimeExecutionError(
            "admit fixed physical layout failed: "
            f"status={status}, requested_slice={layout.slice_bytes}, "
            f"allocated={int(pool.allocated_bytes)}, "
            f"free={int(pool.free_bytes)}, "
            f"free_prefix={int(pool.free_prefix_bytes)}, "
            "largest_free_range="
            f"{int(pool.largest_free_range_bytes)}, "
            "external_fragmentation="
            f"{int(pool.external_fragmentation_bytes)}, "
            f"live_allocations={int(pool.live_allocations)}, "
            f"pool_capacity={int(pool.capacity_bytes)}"
            f"{describe_pool_occupants(bridge)}"
        )
    bridge._fixed_layout_installed = True


def admit_initial_actions(
    bridge: RuntimeBridge,
    actions: tuple[MemoryAction, ...],
    *,
    task_number: int,
    action_trace_labels: tuple[str, ...] | None = None,
) -> int:
    """Admit one reusable initial-placement action batch."""

    labels = action_labels(actions, action_trace_labels)
    action_pairs = _runtime_actions(bridge, actions, labels)
    expected = tuple(
        (action.alias_group_id, action.kind) for action, _label in action_pairs
    )
    existing = bridge._admitted_action_batches.get(task_number)
    if existing is not None:
        existing_handle, admitted = existing
        if admitted != expected:
            raise RuntimeExecutionError(
                "initial action admission changed for an existing task: "
                f"task={task_number}, expected={admitted}, observed={expected}"
            )
        return existing_handle
    for action, _label in action_pairs:
        bridge.objects.register_placeholder(action.alias_group_id)
    bridge.objects.bind(action.alias_group_id for action, _label in action_pairs)
    encoded_labels = tuple(
        label.encode("utf-8") if label else None for _action, label in action_pairs
    )
    runtime_actions = (RuntimeAction * len(action_pairs))(
        *(
            runtime_action(
                plan_local_id(action.alias_group_id, "alias_"),
                ACTION_KIND[action.kind],
                trace_label=encoded,
            )
            for (action, _label), encoded in zip(
                action_pairs, encoded_labels, strict=True
            )
        )
    )
    action_batch_handle = ctypes.c_size_t()
    bridge.require(
        bridge.runtime_library.shadowspill_plan_admit_action_batch(
            bridge.plan_handle,
            task_number,
            runtime_actions if action_pairs else None,
            len(action_pairs),
            ctypes.byref(action_batch_handle),
        ),
        f"admit initial action batch {task_number}",
    )
    if action_batch_handle.value == 0:
        raise RuntimeExecutionError(
            "runtime returned a null initial action batch handle"
        )
    resolved = int(action_batch_handle.value)
    bridge._admitted_action_batches[task_number] = (resolved, expected)
    return resolved


def admit_caller_acquisition(bridge: RuntimeBridge, alias_ids: tuple[str, ...]) -> int:
    """Admit one immutable ordered public-object acquisition handle."""

    runtime_aliases = tuple(
        alias_id for alias_id in alias_ids if bridge.objects.requires_storage(alias_id)
    )
    existing = bridge._admitted_acquisitions.get(runtime_aliases)
    if existing is not None:
        return existing
    if not runtime_aliases:
        return 0
    for alias_id in dict.fromkeys(runtime_aliases):
        bridge.objects.register_placeholder(alias_id)
    bridge.objects.bind(runtime_aliases)
    identifiers = (ctypes.c_uint64 * len(runtime_aliases))(
        *(plan_local_id(value, "alias_") for value in runtime_aliases)
    )
    handle = ctypes.c_size_t()
    bridge.require(
        bridge.runtime_library.shadowspill_plan_admit_object_acquisition(
            bridge.plan_handle,
            identifiers,
            len(runtime_aliases),
            ctypes.byref(handle),
        ),
        "admit caller output acquisition",
    )
    if handle.value == 0:
        raise RuntimeExecutionError(
            "runtime returned a null caller output acquisition handle"
        )
    bridge._admitted_acquisitions[runtime_aliases] = int(handle.value)
    return int(handle.value)


def seal_fixed_layout(bridge: RuntimeBridge) -> None:
    """Resolve the installed certificate after execution admission."""

    if not bridge._fixed_layout_installed:
        raise RuntimeExecutionError("no fixed physical layout was admitted")
    bridge.require(
        bridge.runtime_library.shadowspill_plan_seal_fixed_layout(bridge.plan_handle),
        "seal fixed physical layout",
    )


def clear_tasks(bridge: RuntimeBridge) -> None:
    """Discard immutable task records and their fixed layout."""

    bridge.require(
        bridge.runtime_library.shadowspill_plan_clear_tasks(bridge.plan_handle),
        "clear plan tasks",
    )
    bridge._admitted_task_handles.clear()
    bridge._admitted_action_batches.clear()
    bridge._admitted_acquisitions.clear()
    bridge._fixed_layout_installed = False


def _runtime_inputs(
    bridge: RuntimeBridge, input_alias_ids: tuple[str, ...]
) -> tuple[str, ...]:
    return tuple(
        alias_id
        for alias_id in input_alias_ids
        if bridge.objects.requires_storage(alias_id)
    )


def _runtime_mutations(
    bridge: RuntimeBridge,
    mutations: tuple[MutationSpec, ...],
) -> tuple[MutationSpec, ...]:
    return tuple(
        mutation
        for mutation in mutations
        if bridge.objects.requires_storage(
            bridge.objects.alias_for_object(mutation.object_id)
        )
    )


def _runtime_actions(
    bridge: RuntimeBridge,
    actions: tuple[MemoryAction, ...],
    labels: tuple[str, ...],
) -> tuple[tuple[MemoryAction, str], ...]:
    return tuple(
        (action, label)
        for action, label in zip(actions, labels, strict=True)
        if bridge.objects.requires_storage(action.alias_group_id)
    )


def _referenced_aliases(
    bridge: RuntimeBridge,
    inputs: tuple[str, ...],
    mutations: tuple[MutationSpec, ...],
    publications: tuple[TaskPublication, ...],
    actions: tuple[tuple[MemoryAction, str], ...],
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                *inputs,
                *(
                    bridge.objects.alias_for_object(item.object_id)
                    for item in mutations
                ),
                *(item.alias_id for item in publications),
                *(action.alias_group_id for action, _label in actions),
            )
        )
    )


def encode_task(
    bridge: RuntimeBridge,
    task: TaskSpec,
    inputs: tuple[str, ...],
    mutations: tuple[MutationSpec, ...],
    publications: tuple[TaskPublication, ...],
    actions: tuple[tuple[MemoryAction, str], ...],
    memory_envelope: TaskMemoryEnvelope,
    trace_label: str,
) -> EncodedTask:
    input_ids = (ctypes.c_uint64 * len(inputs))(
        *(plan_local_id(value, "alias_") for value in inputs)
    )
    updates = (ObjectUpdate * len(mutations))(
        *(
            ObjectUpdate(
                plan_local_id(
                    bridge.objects.alias_for_object(item.object_id), "alias_"
                ),
                item.version_delta,
            )
            for item in mutations
        )
    )
    publication_values = (TaskPublicationDescription * len(publications))(
        *(
            TaskPublicationDescription(
                object_id=plan_local_id(item.alias_id, "alias_"),
                kind=1 if item.replace_lease else 0,
            )
            for item in publications
        )
    )
    labels = tuple(label.encode("utf-8") if label else None for _, label in actions)
    action_values = (RuntimeAction * len(actions))(
        *(
            runtime_action(
                plan_local_id(action.alias_group_id, "alias_"),
                ACTION_KIND[action.kind],
                trace_label=label,
            )
            for (action, _text), label in zip(actions, labels, strict=True)
        )
    )
    allocation_contract = memory_envelope.allocation_contract
    contract_steps = () if allocation_contract is None else allocation_contract.steps
    contract_values = (TaskAllocationContractStep * len(contract_steps))(
        *(
            TaskAllocationContractStep(
                allocation_ordinal=step.allocation_ordinal,
                requested_bytes=step.requested_bytes,
                charged_bytes=step.charged_bytes,
                alignment_bytes=step.alignment_bytes,
                operation=0 if step.operation.value == "allocate" else 1,
                required=step.required,
            )
            for step in contract_steps
        )
    )
    encoded_task_label = trace_label.encode("utf-8")
    description = TaskDescription(
        task_id=plan_local_id(task.task_id, "task_"),
        trace_label=encoded_task_label,
        input_object_ids=input_ids if inputs else None,
        input_count=len(inputs),
        updates=updates if mutations else None,
        update_count=len(mutations),
        publications=publication_values if publications else None,
        publication_count=len(publications),
        actions=action_values if actions else None,
        action_count=len(actions),
        allocation_contract_steps=contract_values if contract_steps else None,
        allocation_contract_step_count=len(contract_steps),
        enforce_allocation_contract=allocation_contract is not None,
        maximum_requested_allocation_bytes=(
            memory_envelope.maximum_requested_allocation_bytes
        ),
        maximum_charged_allocation_bytes=(
            memory_envelope.maximum_charged_allocation_bytes
        ),
        live_requested_allocation_limit_bytes=(
            memory_envelope.live_requested_allocation_limit_bytes
        ),
        live_charged_allocation_limit_bytes=(
            memory_envelope.live_charged_allocation_limit_bytes
        ),
        dynamic_scratch_maximum_allocation_bytes=(
            memory_envelope.dynamic_scratch_maximum_allocation_bytes
        ),
        dynamic_scratch_live_limit_bytes=(
            memory_envelope.dynamic_scratch_live_limit_bytes
        ),
    )
    return EncodedTask(
        description,
        encoded_task_label,
        input_ids,
        updates,
        publication_values,
        action_values,
        contract_values,
        labels,
    )


__all__ = [
    "EncodedTask",
    "admit_caller_acquisition",
    "admit_fixed_layout",
    "admit_initial_actions",
    "admit_task",
    "clear_tasks",
    "encode_task",
    "seal_fixed_layout",
]
