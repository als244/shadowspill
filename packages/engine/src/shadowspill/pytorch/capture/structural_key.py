"""Conservative in-process keys for explicit tensor graph analyses.

Only ordinary fixed tensor geometry, input storage spans/aliases, dispatcher schemas
and literal arguments are cacheable. No live tensors, nodes or module state are
retained in keys. Unsupported representations take the uncached path.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping
from typing import Any

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx import GraphModule, Node
from torch.fx.immutable_collections import immutable_dict, immutable_list

from .live_storage import live_storage_identity


class _Uncacheable(Exception):
    pass


_MISSING = object()


def _value_key(value: Any, nodes: Mapping[Node, int], *, tensors: bool) -> Any:
    if isinstance(value, Node):
        return (Node, nodes[value])
    if isinstance(value, torch.Tensor):
        if not tensors or type(value) not in (
            torch.Tensor,
            torch.nn.Parameter,
            FakeTensor,
        ):
            raise _Uncacheable
        if value.layout is not torch.strided:
            raise _Uncacheable
        shape, stride, offset = (
            tuple(value.shape),
            tuple(value.stride()),
            value.storage_offset(),
        )
        if any(type(item) is not int for item in (*shape, *stride, offset)):
            raise _Uncacheable
        return (
            torch.Tensor,
            shape,
            stride,
            offset,
            value.dtype,
            value.device,
            value.requires_grad,
            value.is_conj(),
            value.is_neg(),
        )
    if value is _MISSING:
        return _MISSING
    if value is None or type(value) in (bool, int, str):
        return (type(value), value)
    if type(value) is float:
        return (float, value.hex())
    if type(value) in (tuple, list, immutable_list):
        return (
            tuple if isinstance(value, tuple) else list,
            tuple(_value_key(v, nodes, tensors=tensors) for v in value),
        )
    if type(value) in (dict, immutable_dict):
        return (
            dict,
            tuple(
                (
                    _value_key(k, nodes, tensors=False),
                    _value_key(v, nodes, tensors=tensors),
                )
                for k, v in value.items()
            ),
        )
    if type(value) in (torch.dtype, torch.device, torch.layout, torch.memory_format):
        return (type(value), value)
    raise _Uncacheable


def graph_input_key(
    graph: GraphModule,
    inputs: tuple[object, ...],
) -> tuple[Any, ...] | None:
    """Return a conservative in-process key; never identify real tensor values."""
    nodes = {node: index for index, node in enumerate(graph.graph.nodes)}
    if any(
        node.op not in {"placeholder", "output", "call_function"}
        or (node.op == "call_function"
        and not (
            isinstance(node.target, torch._ops.OpOverload)
            or node.target is operator.getitem
        ))
        for node in nodes
    ):
        return None
    aliases: dict[int, int] = {}
    arguments = []
    try:
        for value in inputs:
            # The alias catalog below covers flat tensor arguments. Containers
            # may carry other literals, but hidden tensor aliases need a richer
            # input contract before their analysis can be reused.
            item = _value_key(value, nodes, tensors=isinstance(value, torch.Tensor))
            storage_key = None
            if isinstance(value, torch.Tensor):
                size = value.untyped_storage().nbytes()
                if type(size) is not int:
                    raise _Uncacheable
                storage_key = (
                    aliases.setdefault(live_storage_identity(value), len(aliases)),
                    size,
                )
            arguments.append((item, storage_key))
        operations = tuple(
            (
                node.op,
                node.target if node.op == "call_function" else None,
                str(getattr(node.target, "_schema", "")),
                _value_key(node.args, nodes, tensors=False),
                _value_key(node.kwargs, nodes, tensors=False),
                _value_key(node.meta.get("val", _MISSING), nodes, tensors=True),
            )
            for node in nodes
        )
    except _Uncacheable:
        return None
    return operations, tuple(arguments), torch.get_default_dtype()
