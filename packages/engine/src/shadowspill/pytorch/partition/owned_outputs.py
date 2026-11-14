"""Keep alias-output VJPs inside the explicit stage backward contract."""

from __future__ import annotations

from collections import Counter

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

from shadowspill.pytorch.capture.storage import capture_task_storage_contract
from shadowspill.pytorch.representations import is_wrapper
from shadowspill.task.storage import StorageRootKind

from .split import SplitExportGraph, observe_split_root


def own_differentiable_aliases(
    split: SplitExportGraph, inputs: tuple[object, ...]
) -> SplitExportGraph:
    """Materialize differentiable input/view aliases and reobserve their users.

    AOT's autograd wrapper differentiates alias outputs outside the explicit
    backward graph. Independent roots retain those cotangents in the graph we
    serialize. Reobserving the split propagates the actual boundary strides to
    downstream stages before any graph pair or storage contract is captured.
    """
    changed = False
    for record in split.stages:
        graph = record.graph_module.graph
        output = next(node for node in graph.nodes if node.op == "output")
        leaves, spec = tree_flatten(output.args[0])
        values, _ = tree_flatten(record.output)
        eligible = {
            i
            for i, value in enumerate(values)
            if isinstance(value, torch.Tensor)
            and value.requires_grad
            and not is_wrapper(value)
        }
        if not eligible:
            continue
        contract = capture_task_storage_contract(record.graph_module, record.inputs)
        counts = Counter(
            view.root_id
            for view in contract.output_views
            if view.leaf_index in eligible
        )
        for view in contract.output_views:
            if view.leaf_index not in eligible:
                continue
            root = contract.roots[view.root_id]
            if root.kind is not StorageRootKind.INPUT and counts[root.root_id] == 1:
                continue
            with graph.inserting_before(output):
                node = graph.call_function(
                    torch.ops.aten.clone.default, (leaves[view.leaf_index],)
                )
            node.meta = dict(leaves[view.leaf_index].meta)
            leaves[view.leaf_index] = node
            changed = True
        output.args = (tree_unflatten(leaves, spec),)
        graph.lint()
        record.graph_module.recompile()
    return observe_split_root(split.root, inputs) if changed else split
