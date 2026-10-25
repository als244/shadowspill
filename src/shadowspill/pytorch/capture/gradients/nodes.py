"""Geometry metadata and recognized value-preserving graph operations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, cast

import torch
from torch.fx.passes.shape_prop import _extract_tensor_metadata

# Operations that only convert a tensor's dtype.
_CONVERTING = frozenset(
    {
        torch.ops.prims.convert_element_type.default,
        torch.ops.aten._to_copy.default,
    }
)


# GEMM overloads that accept an explicit accumulation/output dtype.
_WRITING_AT_DTYPE: dict[object, torch._ops.OpOverload] = {
    torch.ops.aten.mm.default: torch.ops.aten.mm.dtype,
    torch.ops.aten.bmm.default: torch.ops.aten.bmm.dtype,
    torch.ops.aten.addmm.default: torch.ops.aten.addmm.dtype,
    torch.ops.aten.baddbmm.default: torch.ops.aten.baddbmm.dtype,
}


# Views and copies preserve values between a gradient producer and its output.
_MOVING_ONLY = frozenset(
    {
        torch.ops.aten.alias.default,
        torch.ops.aten.clone.default,
        torch.ops.aten.permute.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.squeeze.default,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.squeeze.dims,
        torch.ops.aten.t.default,
        torch.ops.aten.transpose.int,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.view.default,
        torch.ops.aten._unsafe_view.default,
    }
)


# These multiplies can add directly into an existing gradient.
_ADDS_INTO = frozenset(
    {
        torch.ops.aten.mm.default,
        torch.ops.aten.mm.dtype,
        torch.ops.aten.bmm.default,
        torch.ops.aten.bmm.dtype,
    }
)


def _undo_move(
    node: torch.fx.Node,
) -> tuple[Callable[..., Any], tuple[Any, ...]] | None:
    """The view that undoes the move ``node`` makes: the operation and the
    arguments after the tensor, or ``None`` for a copy, which moves nothing."""

    target = node.target
    if target is torch.ops.aten.clone.default:
        return None
    if target in (torch.ops.aten.t.default, torch.ops.aten.transpose.int):
        return cast(Callable[..., Any], target), tuple(node.args[1:])
    if target is torch.ops.aten.permute.default:
        axes = cast(Sequence[int], node.args[1])
        order = [axis % len(axes) for axis in axes]
        return torch.ops.aten.permute.default, (
            [order.index(axis) for axis in range(len(order))],
        )
    source = cast(torch.Tensor, cast(torch.fx.Node, node.args[0]).meta["val"])
    return torch.ops.aten.view.default, (list(source.shape),)


def _record_value(node: torch.fx.Node, value: torch.Tensor) -> None:
    """Record what ``node`` now computes, as graph capture records it."""

    node.meta["val"] = value
    node.meta["tensor_meta"] = _extract_tensor_metadata(value)
