"""Reuse split geometry after the current occurrence's cuts are resolved."""

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx import GraphModule, Node
from torch.fx._lazy_graph_module import _use_lazy_graph_module
from torch.fx.passes.split_module import split_module
from torch.utils._pytree import tree_flatten

from shadowspill.pytorch.capture.structural_key import graph_input_key
from shadowspill.pytorch.capture.torch_deprecations import copy_graph_module
from shadowspill.pytorch.partition.sources import stage_value_source
from shadowspill.pytorch.partition.split import (
    SplitExportGraph,
    observe_split_root,
    rebind_split_records,
)

_SPLIT_SOURCE_INDEX = "shadowspill_split_source_index"


def _rebindable(values: object) -> bool:
    leaves, _ = tree_flatten(values)
    return all(
        (
            type(value) is FakeTensor
            and value.layout is torch.strided
            and value.constant is None
            and not value.is_conj()
            and not value.is_neg()
            and all(
                type(item) is int
                for item in (
                    *value.shape,
                    *value.stride(),
                    value.storage_offset(),
                    value.untyped_storage().nbytes(),
                )
            )
        )
        if isinstance(value, torch.Tensor)
        else type(value) in (type(None), bool, int, float, str)
        for value in leaves
    )


class SplitObservations:
    """Bounded fake storage templates; never retain concrete tensor contents."""

    def __init__(self, max_entries: int) -> None:
        self.max_entries = max_entries
        self.entries: OrderedDict[
            tuple[Any, ...], tuple[SplitExportGraph, tuple[object, ...]]
        ] = OrderedDict()
        self.hits = self.misses = self.bypasses = 0

    def observe(
        self,
        graph: GraphModule,
        assignments: Mapping[Node, int],
        inputs: tuple[object, ...],
    ) -> SplitExportGraph:
        signature = graph_input_key(graph, inputs) if _rebindable(inputs) else None
        if signature is None:
            self.bypasses += 1
            return split_and_observe(graph, assignments, inputs)
        key = (
            signature,
            tuple(
                (node.name, str(node.target), assignments.get(node))
                for node in graph.graph.nodes
            ),
        )
        previous = self.entries.get(key)
        if previous is not None:
            template, arguments = previous
            rebound = rebind_split_records(
                template.root, template.stages, arguments, inputs
            )
            if rebound is not None:
                self.hits += 1
                self.entries.move_to_end(key)
                # Geometry is structural. Graph/source metadata belongs to the
                # current occurrence and is still visible to every inner policy.
                return replace(
                    rebound,
                    stages=tuple(
                        replace(
                            record,
                            graph_module=_bind_stage(record.graph_module, graph),
                        )
                        for record in rebound.stages
                    ),
                )
        self.misses += 1
        for index, node in enumerate(graph.graph.nodes):
            node.meta[_SPLIT_SOURCE_INDEX] = index
        result = split_and_observe(graph, assignments, inputs)
        # Metadata-changing input mutations cannot be rebound as ordinary
        # writes to a fixed storage contract.
        if (
            _rebindable(
                tuple((record.inputs, record.output) for record in result.stages)
            )
            and graph_input_key(graph, inputs) == signature
        ):
            self.entries[key] = (result, inputs)
            self.entries.move_to_end(key)
            if len(self.entries) > self.max_entries:
                self.entries.popitem(last=False)
        return result


def split_and_observe(
    graph: GraphModule,
    assignments: Mapping[Node, int],
    inputs: tuple[object, ...],
) -> SplitExportGraph:
    # Most rebound copies are only inspected, then deduplicated before
    # compilation. Keep their wrappers lazy; execution and .code inspection
    # still generate current code through PyTorch's ordinary lazy dispatch.
    with _use_lazy_graph_module(True):
        root = split_module(
            graph, graph, lambda node: assignments[node], keep_original_order=True
        )
    return observe_split_root(root, inputs)


def _bind_stage(template: GraphModule, source: GraphModule) -> GraphModule:
    """Splitting preserves node metadata; substitute the current source copy."""
    graph = copy_graph_module(template)
    nodes = tuple(source.graph.nodes)
    for node in graph.graph.nodes:
        index = node.meta.get(_SPLIT_SOURCE_INDEX)
        if index is not None:
            node.meta = dict(nodes[index].meta)
    return graph


def split_output_values(split: SplitExportGraph, inputs: tuple[Any, ...]) -> list[Any]:
    """Read already observed outputs instead of executing the complete task again."""
    output = next(node for node in split.root.graph.nodes if node.op == "output")
    leaves, _ = tree_flatten(output.args[0])
    stage_outputs = [tree_flatten(record.output)[0] for record in split.stages]
    values = []
    for leaf in leaves:
        if not isinstance(leaf, Node):
            values.append(leaf)
            continue
        source = stage_value_source(
            leaf,
            placeholder_index=split.placeholder_index,
            stage_index_by_node=split.stage_index_by_node,
        )
        if source.root_input_index is not None:
            values.append(inputs[source.root_input_index])
        else:
            assert source.producer_stage_index is not None
            assert source.producer_output_index is not None
            values.append(
                stage_outputs[source.producer_stage_index][source.producer_output_index]
            )
    return values
