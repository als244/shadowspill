"""Caller-owned output handoff and the small training result record."""

from dataclasses import dataclass, field
from typing import Any

from torch.utils._pytree import tree_unflatten, treespec_loads

from shadowspill.artifacts.examples import output_groups, single_forward
from shadowspill.pytorch.runtime_adapter.outputs import (
    acquire_for_caller,
    transfer_outputs_to_caller,
)

from .storage import Storage


@dataclass(frozen=True)
class TrainingResult:
    """The losses and metrics of one training run, per example.

    The handle `Executor.run` returns resolves to one for a step,
    forward/backward or optimizer-only Plan; a forward Plan's resolves to
    the model's outputs instead.

    Attributes
    ----------
    losses : tuple
        Each example's loss as the objective returned it, a scalar tensor the
        caller owns, in example order. Empty for an optimizer-only Plan.
    metrics : tuple
        Each example's metrics, in the structure the objective returned.
    parameter_metrics : dict
        When the run collects parameter norms, each trained parameter's
        ``"param_norm"``, the norm of its weights before the update, and
        ``"grad_norm"``, the norm of the gradient the optimizer applied, over
        all workers, as host float32 scalar tensors. Empty otherwise.
    """

    losses: tuple[Any, ...]
    metrics: tuple[Any, ...]
    parameter_metrics: dict[str, Any] = field(default_factory=dict)


def collect_outputs(state: Storage, acquisition: int) -> Any:
    bindings = state.plan.program.bindings
    groups = output_groups(bindings)
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
    state.host_values.values.clear()
    if bindings["kind"] == "forward":
        tree = treespec_loads(bindings["output_tree"])
        for values in outputs:
            for index, value in bindings["output_literals"]:
                values[index] = value
        restored = tuple(tree_unflatten(values, tree) for values in outputs)
        return restored[0] if single_forward(bindings) else restored
    schema, metrics = bindings["objective_schema"], []
    for example, values in enumerate(outputs):
        leaves = [None] * schema["leaf_count"]
        for position, value in schema["static_leaves"][example]:
            leaves[position] = value
        for position, value in zip(schema["tensor_positions"], values[1:], strict=True):
            leaves[position] = value
        metrics.append(tree_unflatten(leaves, treespec_loads(schema["tree"])))
    return TrainingResult(tuple(values[0] for values in outputs), tuple(metrics))
