"""Run one saved callable between exact acquisition/publication boundaries."""

import time
from typing import TYPE_CHECKING, Any

import torch
from torch.utils._pytree import tree_flatten

from shadowspill.ir import MemoryActionKind
from shadowspill.pytorch.runtime_adapter.boundaries import (
    PublishedStorage,
    after_task_and_update,
    before_task_and_acquire,
    wait_task_allocations,
)
from shadowspill.pytorch.runtime_adapter.replacements import ReplacementStorageViews
from shadowspill.runtime.plan import abort_task
from shadowspill.runtime.plan.annotations import TaskBoundaryAnnotations

from .admission import BoundTask
from .controls import Controls
from .storage import Storage

if TYPE_CHECKING:
    from shadowspill.diagnostics.timing import ArmedTaskTiming

    from .observations.collector import ParameterNorms
    from .tracing import InvocationTrace


def dispatch(
    bound: BoundTask,
    state: Storage,
    controls: Controls,
    observations: "ParameterNorms | None" = None,
    annotations: TaskBoundaryAnnotations | None = None,
    trace: "InvocationTrace | None" = None,
) -> None:
    if annotations is None:
        _dispatch(bound, state, controls, observations, trace)
        return
    name = state.plan.program.bindings.get("task_names", {}).get(
        bound.entry.task_id, bound.entry.task_id
    )
    with annotations.range(f"shadowspill.task.{name}"):
        _dispatch(bound, state, controls, observations, trace)


def _dispatch(
    bound: BoundTask,
    state: Storage,
    controls: Controls,
    observations: "ParameterNorms | None",
    trace: "InvocationTrace | None",
) -> None:
    bridge, device = state.bridge, state.device.index
    timing = None if trace is None else trace.timing
    task = None if timing is None else timing.begin_task(bound.entry)
    stream = torch.cuda.current_stream(state.device) if timing is not None else None
    if timing is not None:
        timing.record_task_readiness(task, stream)
    tensors = [
        state.owner(a) for a in bound.inputs if bridge.objects.requires_storage(a)
    ]
    before_task_and_acquire(bridge, bound.handle, device, tensors)
    try:
        if timing is not None:
            timing.record_task_inputs_ready(task, stream)
        wait_task_allocations(bridge, bound.handle, device)
        if timing is not None:
            assert trace is not None
            if bound.entry.task_id == trace.first:
                timing.record_compute_start(stream)
            timing.record_task_start(task, stream)
        _run(bound, state, controls, observations, trace, task, stream)
    except BaseException as error:
        error.add_note(f"while executing task {bound.entry.task_id}")
        abort_task(bridge, bound.handle)
        raise
    finally:
        if timing is not None:
            timing.finish_task(task)


def _run(
    bound: BoundTask,
    state: Storage,
    controls: Controls,
    observations: "ParameterNorms | None",
    trace: "InvocationTrace | None",
    task: "ArmedTaskTiming | None",
    stream: torch.cuda.Stream | None,
) -> None:
    description, entry = bound.description, bound.entry
    arguments: list[Any] = [None] * description.get("count", 0)
    for index, value in description.get("static_values", ()):
        arguments[index] = value
    geometries = dict(
        zip(
            description.get("tensor_positions", ()),
            description.get("tensor_geometry", ()),
            strict=True,
        )
    )
    for slot in entry.input_slots:
        arguments[slot.leaf_index] = state.argument(
            slot.object_id, geometries[slot.leaf_index]
        )
    for position, name in bound.host_inputs:
        arguments[position] = controls.host[name]
    with torch.no_grad():
        if observations is not None:
            # Allocation-free reads on the task's own stream complete before
            # its update/publication can overwrite or retire these inputs.
            observations.observe(bound, arguments)
        if task is not None:
            task.before_task_exit_ns = time.perf_counter_ns()
        leaves, _ = tree_flatten(
            () if bound.function is None else bound.function(*arguments)
        )
        if task is not None:
            assert trace is not None
            task.after_task_enter_ns = time.perf_counter_ns()
            trace.timing.record_task_end(task, stream)
            if bound.entry.task_id == trace.last:
                trace.timing.record_compute_end(stream)
    # A returned input alias is an in-place update. Extra eager reductions would
    # be unprofiled work, so refuse those until build packages that computation.
    for slot in entry.options.contribution_slots:
        alias = state.aliases[slot.object_id]
        value = leaves[slot.leaf_index]
        if (
            alias in bound.inputs
            and value.untyped_storage()._cdata
            != state.owner(alias).untyped_storage()._cdata
        ):
            raise NotImplementedError(
                "gradient combination must be part of a compiled task"
            )
    adopted, replacements = [], []
    for ordinal, publication in enumerate(bound.publications):
        binding = next(
            b for b in bound.outputs if b.alias_group_id == publication.alias_id
        )
        value = leaves[binding.leaf_index]
        adopted.append(PublishedStorage(value, publication.alias_id, ordinal))
        if publication.replace_lease and publication.alias_id in state.owners:
            replacements.append(
                ReplacementStorageViews(
                    publication.alias_id, (state.owner(publication.alias_id),)
                )
            )
        else:
            state.owners[publication.alias_id] = value
    for slot in (*entry.output_slots, *entry.options.contribution_slots):
        value = leaves[slot.leaf_index]
        state.outputs[slot.object_id] = value
        if isinstance(value, torch.Tensor) and value.device.type == "cpu":
            state.host_values.values[slot.object_id] = value
    state.host_values.finish(entry.task_id)
    handoff_sources = {
        b.source_alias_group_id for b in bound.outputs if b.source_alias_group_id
    }
    release = {
        a.alias_group_id
        for a in bound.actions
        if a.kind in {MemoryActionKind.RELEASE, MemoryActionKind.EVICT}
    }
    dematerialized = [state.owner(a) for a in release - handoff_sources]
    # Keep only declared public results, not transient function returns, alive
    # across the task boundary. Program inputs are reconstructed from metadata.
    public = state.plan.program.bindings["public_outputs"]
    if state.plan.program.bindings["kind"] != "forward":
        public = [o for example in public for o in example]
    state.outputs = {o: value for o, value in state.outputs.items() if o in public}
    del leaves, arguments
    after_task_and_update(
        state.bridge,
        bound.handle,
        state.device.index,
        adopted,
        tuple(range(len(adopted))),
        dematerialized,
        replacements=replacements,
    )
    for alias in handoff_sources:
        state.owners.pop(alias, None)
