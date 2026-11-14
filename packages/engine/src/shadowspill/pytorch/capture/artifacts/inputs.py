"""Normalize tensor inputs, static arguments and explicit mutation positions."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
from torch.fx import GraphModule
from torch.fx.node import Node, map_arg

from shadowspill.pytorch.capture.live_storage import live_storage_identity
from shadowspill.pytorch.capture.storage import (
    ExplicitMutation,
)
from shadowspill.task.inputs import TaskInputRole

from ..torch_deprecations import copy_graph_module
from .geometry import TaskInputProvenance


def _tensor_arguments(values: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    return tuple(value for value in values if isinstance(value, torch.Tensor))


def _tensor_alias_groups(values: tuple[torch.Tensor, ...]) -> tuple[int, ...]:
    group_by_storage: dict[int, int] = {}
    return tuple(
        group_by_storage.setdefault(
            live_storage_identity(value),
            len(group_by_storage),
        )
        for value in values
    )


def _compact_mutations(
    mutations: tuple[ExplicitMutation, ...],
    tensor_positions: tuple[int, ...],
) -> tuple[ExplicitMutation, ...]:
    compact_position = {
        original: compact for compact, original in enumerate(tensor_positions)
    }
    compact: list[ExplicitMutation] = []
    for mutation in mutations:
        try:
            input_position = compact_position[mutation.input_position]
        except KeyError as exc:
            raise ValueError(
                "functional mutation target was specialized as static"
            ) from exc
        compact.append(
            ExplicitMutation(
                input_position,
                mutation.output_leaf_index,
                mutation.target,
            )
        )
    return tuple(compact)


def _normalize_input_provenance(
    graph_module: GraphModule,
    original_inputs: tuple[object, ...],
    tensor_positions: tuple[int, ...],
    supplied: tuple[TaskInputProvenance, ...] | None,
) -> tuple[TaskInputProvenance, ...]:
    """Attach compact tensor roles and static FX consumer descriptions."""

    if supplied is None:
        original = tuple(
            TaskInputProvenance(TaskInputRole.ANONYMOUS) for _value in original_inputs
        )
    else:
        if len(supplied) != len(original_inputs):
            raise ValueError("task input provenance must match argument arity")
        original = supplied
    compact = tuple(original[position] for position in tensor_positions)
    placeholders = tuple(
        node for node in graph_module.graph.nodes if node.op == "placeholder"
    )
    if len(placeholders) != len(compact):
        raise ValueError("specialized task placeholders differ from tensor inputs")
    result: list[TaskInputProvenance] = []
    for item, placeholder in zip(compact, placeholders, strict=True):
        consumers = tuple(
            sorted(
                {
                    str(user.target)
                    for user in placeholder.users
                    if user.op in {"call_function", "call_method", "call_module"}
                }
            )
        )
        result.append(replace(item, consumer_targets=consumers))
    return tuple(result)


def _specialize_static_inputs(
    graph_module: GraphModule, example_inputs: tuple[object, ...]
) -> tuple[GraphModule, tuple[object, ...], tuple[int, ...]]:
    """Replace guarded non-tensor placeholders with their captured constants."""

    placeholders = tuple(
        node for node in graph_module.graph.nodes if node.op == "placeholder"
    )
    if len(placeholders) != len(example_inputs):
        raise ValueError("graph placeholder count differs from example arguments")
    tensor_positions = tuple(
        index
        for index, value in enumerate(example_inputs)
        if isinstance(value, torch.Tensor)
    )
    if len(tensor_positions) == len(example_inputs):
        return graph_module, example_inputs, tensor_positions
    specialized = copy_graph_module(graph_module)
    specialized_placeholders = tuple(
        node for node in specialized.graph.nodes if node.op == "placeholder"
    )
    for placeholder, value in zip(
        specialized_placeholders, example_inputs, strict=True
    ):
        if isinstance(value, torch.Tensor):
            continue

        def replace(
            node: Node,
            target: Node = placeholder,
            replacement: object = value,
        ) -> Any:
            return replacement if node is target else node

        for user in tuple(placeholder.users):
            user.args = map_arg(user.args, replace)
            user.kwargs = map_arg(user.kwargs, replace)
        specialized.graph.erase_node(placeholder)
    specialized.graph.lint()
    specialized.recompile()
    return (
        specialized,
        tuple(value for value in example_inputs if isinstance(value, torch.Tensor)),
        tensor_positions,
    )
