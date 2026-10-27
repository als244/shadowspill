"""Bind training outputs to canonical storage objects."""

from __future__ import annotations

from shadowspill.build.lowering.geometry import serialized_dtype_role
from shadowspill.errors import CaptureError
from shadowspill.ir import ObjectRole, Persistence
from shadowspill.pytorch.capture.artifacts import AotGraphPair
from shadowspill.pytorch.capture.storage import TaskStorageContract
from shadowspill.pytorch.graph_pairs import (
    saved_value_footprint,
)
from shadowspill.task.geometry import TaskStorageGeometry
from shadowspill.task.slots import ObjectSlot

from ..objects import ObjectCatalog
from ..task_binding import (
    TaskBindingResolver,
    TaskStorageHandoff,
)


def bind_forward_outputs(
    pair: AotGraphPair,
    forward_inputs: tuple[ObjectSlot, ...],
    canonical_outputs: tuple[str, ...],
    inventory: ObjectCatalog,
    compiled_layout: TaskStorageGeometry,
    storage_contract: TaskStorageContract,
    *,
    problem: str,
) -> tuple[
    tuple[ObjectSlot, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[int, ...],
    tuple[TaskStorageHandoff, ...],
]:
    public_count = pair.forward.output_count - pair.saved_value_count
    if public_count != len(canonical_outputs):
        raise CaptureError("stage boundary output count changed across AOT capture")
    slots: list[ObjectSlot] = []
    saved_internal_object_ids: list[str] = []
    if compiled_layout.contract_digest != storage_contract.compatibility_digest:
        raise CaptureError(f"{problem}: compiled layout belongs to another contract")
    resolver = TaskBindingResolver(
        inventory,
        pair.forward,
        forward_inputs,
        compiled_layout,
        storage_contract=storage_contract,
    )
    internal_root_ids = frozenset(
        saved_value_footprint(pair, storage_contract).internal_root_ids
    )
    root_by_leaf = {
        item.leaf_index: item.root_id for item in storage_contract.output_views
    }
    view_by_leaf = {item.leaf_index: item for item in storage_contract.output_views}
    for index in range(pair.forward.output_count):
        try:
            role = serialized_dtype_role(
                view_by_leaf[index].dtype,
                continuous_role=ObjectRole.ACTIVATION,
            )
        except KeyError as error:
            raise CaptureError(
                f"{problem}: forward output {index} has no storage view"
            ) from error
        if index < public_count:
            object_id = resolver.bind(
                index,
                role=role,
                persistence=Persistence.STEP,
                canonical_object_id=canonical_outputs[index],
            )
        else:
            object_id = resolver.bind(
                index,
                role=role,
                persistence=Persistence.STEP,
            )
        slots.append(ObjectSlot(index, object_id))
        if index >= public_count and root_by_leaf.get(index) in internal_root_ids:
            saved_internal_object_ids.append(object_id)
    return (
        tuple(slots),
        tuple(dict.fromkeys(saved_internal_object_ids)),
        resolver.mutation_object_ids,
        resolver.replacement_output_leaves,
        resolver.storage_handoffs,
    )
