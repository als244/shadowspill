"""Run one saved callable between exact acquisition/publication boundaries."""

from typing import Any

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

from .admission import BoundTask
from .controls import Controls
from .storage import Storage


def dispatch(bound: BoundTask, state: Storage, controls: Controls) -> None:
    bridge, device = state.bridge, state.device.index
    tensors = [
        state.owner(a) for a in bound.inputs if bridge.objects.requires_storage(a)
    ]
    before_task_and_acquire(bridge, bound.handle, device, tensors)
    try:
        wait_task_allocations(bridge, bound.handle, device)
        _run(bound, state, controls)
    except BaseException:
        abort_task(bridge, bound.handle)
        raise


def _run(bound: BoundTask, state: Storage, controls: Controls) -> None:
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
        leaves, _ = tree_flatten(
            () if bound.function is None else bound.function(*arguments)
        )
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
        state.outputs[slot.object_id] = leaves[slot.leaf_index]
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
