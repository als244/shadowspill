"""Give returned input/state views an owned, profiled snapshot at the boundary."""

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import torch
from torch.fx import Graph, Node

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.export import ExportCapture


def own_public_outputs(capture: ExportCapture) -> None:
    graph_module = capture.exported_program.graph_module
    graph = graph_module.graph
    output = next(n for n in graph.nodes if n.op == "output")
    leaves = list(output.args[0])
    inputs = {
        _storage(n.meta["val"])
        for n in graph.nodes
        if n.op == "placeholder" and isinstance(n.meta.get("val"), torch.Tensor)
    }
    groups: dict[int, list[int]] = {}
    for index in capture.user_output_indices:
        node = leaves[index]
        value = node.meta.get("val") if isinstance(node, Node) else None
        if isinstance(value, torch.Tensor) and _storage(value) in inputs:
            groups.setdefault(_storage(value), []).append(index)
    for indices in groups.values():
        nodes = [leaves[i] for i in indices]
        values = [n.meta["val"] for n in nodes]
        if len({v.dtype for v in values}) != 1:
            raise CaptureError(
                "returned state aliases with mixed dtypes need explicit copies"
            )
        start = min(v.storage_offset() for v in values)
        stop = max(_end(v) for v in values)
        source = nodes[0]
        with graph.inserting_before(output):
            region = _call(
                graph,
                torch.ops.aten.as_strided.default,
                (source, (stop - start,), (1,), start),
            )
            owned = _call(graph, torch.ops.aten.clone.default, (region,))
            views = {}
            for index, node, value in zip(indices, nodes, values, strict=True):
                if node not in views:
                    views[node] = _call(
                        graph,
                        torch.ops.aten.as_strided.default,
                        (
                            owned,
                            tuple(value.shape),
                            tuple(value.stride()),
                            value.storage_offset() - start,
                        ),
                    )
                leaves[index] = views[node]
                specs = capture.exported_program.graph_signature.output_specs
                specs[index] = replace(
                    specs[index], arg=replace(specs[index].arg, name=views[node].name)
                )
    output.args = (tuple(leaves),)
    graph.lint()
    graph_module.recompile()


def _storage(tensor: torch.Tensor) -> int:
    return int(tensor.untyped_storage()._cdata)


def _end(value: torch.Tensor) -> int:
    return int(
        value.storage_offset()
        + (
            0
            if value.numel() == 0
            else 1
            + sum(
                (n - 1) * stride
                for n, stride in zip(value.shape, value.stride(), strict=True)
            )
        )
    )


def _call(graph: Graph, target: Callable[..., Any], args: tuple[Any, ...]) -> Node:
    node = graph.call_function(target, args=args)
    arguments = torch.fx.node.map_arg(args, lambda n: n.meta["val"])
    node.meta["val"] = target(*arguments)
    return node
