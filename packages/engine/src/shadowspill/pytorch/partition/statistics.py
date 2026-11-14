"""Prove that an integer stage output is used only as fixed-shape numeric data.

Unknown operations, indexing, scalar extraction and mutations remain controls.
The proof follows every consumer across stage boundaries through a small set of
ordinary ATen arithmetic operations. It never changes the model's computation.
"""

from __future__ import annotations

from collections import defaultdict

import torch
from torch.fx import Node
from torch.utils._pytree import tree_flatten

from .split import SplitExportGraph

_ARITHMETIC = {
    "aten::_assert_tensor_metadata",
    "aten::add",
    "aten::sub",
    "aten::mul",
    "aten::div",
    "aten::sum",
    "aten::mean",
    "aten::stack",
    "aten::cat",
    "aten::_to_copy",
    "aten::to",
    "aten::clone",
    "aten::detach",
    "aten::view",
    "aten::reshape",
    "aten::unsqueeze",
    "aten::squeeze",
    "aten::expand",
    "aten::permute",
    "aten::transpose",
    "aten::alias",
}


def numeric_outputs(split: SplitExportGraph) -> frozenset[tuple[int, int]]:
    readers: dict[tuple[int, int], list[tuple[int, Node]]] = defaultdict(list)
    outputs = []
    for index, record in enumerate(split.stages):
        graph = record.graph_module.graph
        outputs.append(
            tree_flatten(next(n for n in graph.nodes if n.op == "output").args[0])[0]
        )
        placeholders = tuple(n for n in graph.nodes if n.op == "placeholder")
        for node, source in zip(placeholders, record.input_sources, strict=True):
            if source is not None and source.producer_stage_index is not None:
                assert source.producer_output_index is not None
                readers[
                    source.producer_stage_index, source.producer_output_index
                ].append((index, node))

    # Both the stage DAG and each FX graph are topologically ordered. Work
    # backwards so long forwarding chains do not consume Python's call stack.
    arithmetic: dict[tuple[int, Node], bool] = {}
    for index in reversed(range(len(split.stages))):
        for node in reversed(tuple(split.stages[index].graph_module.graph.nodes)):
            safe = True
            for user in node.users:
                if user.op == "output":
                    safe = all(
                        arithmetic[stage, argument]
                        for position, value in enumerate(outputs[index])
                        if value is node
                        for stage, argument in readers.get((index, position), ())
                    )
                else:
                    schema = getattr(user.target, "_schema", None)
                    safe = (
                        user.op == "call_function"
                        and schema is not None
                        and not schema.is_mutable
                        and schema.name in _ARITHMETIC
                        and arithmetic[index, user]
                    )
                if not safe:
                    break
            arithmetic[index, node] = safe

    result = set()
    for key, consumers in readers.items():
        stage, position = key
        values = tree_flatten(split.stages[stage].output)[0]
        value = values[position]
        if (
            isinstance(value, torch.Tensor)
            and not value.is_floating_point()
            and not value.is_complex()
            and all(arithmetic[item] for item in consumers)
        ):
            result.add(key)
    return frozenset(result)
