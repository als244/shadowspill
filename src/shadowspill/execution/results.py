"""Caller-owned output handoff and the small training result record."""

from dataclasses import dataclass
from typing import Any

from torch.utils._pytree import tree_unflatten, treespec_loads

from shadowspill.pytorch.runtime_adapter.boundaries import (
    acquire_for_caller,
    transfer_outputs_to_caller,
)

from .storage import Storage


@dataclass(frozen=True)
class TrainingResult:
    losses: tuple[Any, ...]
    metrics: tuple[Any, ...]


def collect_outputs(state: Storage, acquisition: int) -> Any:
    bindings = state.plan.program.bindings
    groups = bindings["public_outputs"]
    if bindings["kind"] == "forward":
        groups = [groups]
    public_ids = [o for group in groups for o in group if o is not None]
    tensors = tuple(state.outputs[o] for o in public_ids)
    aliases = tuple(state.aliases[o] for o in public_ids)
    acquired = acquire_for_caller(
        state.bridge, aliases, tensors, acquisition_handle=acquisition
    )
    transfer_outputs_to_caller(
        state.bridge, aliases, tensors, acquired, acquisition_handle=acquisition
    )
    outputs = [
        [None if o is None else state.outputs[o] for o in group] for group in groups
    ]
    for alias in aliases:
        state.owners.pop(alias, None)
    state.outputs.clear()
    if bindings["kind"] == "forward":
        for index, value in bindings["output_literals"]:
            outputs[0][index] = value
        return tree_unflatten(outputs[0], treespec_loads(bindings["output_tree"]))
    schema, metrics = bindings["objective_schema"], []
    for values in outputs:
        leaves = [None] * schema["leaf_count"]
        for position, value in schema["static_leaves"]:
            leaves[position] = value
        for position, value in zip(schema["tensor_positions"], values[1:], strict=True):
            leaves[position] = value
        metrics.append(tree_unflatten(leaves, treespec_loads(schema["tree"])))
    return TrainingResult(tuple(values[0] for values in outputs), tuple(metrics))
