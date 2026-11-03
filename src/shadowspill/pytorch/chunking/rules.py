"""Conservative row-separability rules for explicit tensor graphs."""

from __future__ import annotations

import math
import operator
from collections.abc import Mapping
from typing import Any

import torch
from torch.fx import Node
from torch.utils._pytree import tree_flatten, tree_map

from shadowspill.errors import CaptureError

from .geometry import REDUCED, ChunkAxis, Layout, LocalLayout, reshape_axis
from .policy import AxisFrom, ChunkLocal, OperatorRule

_ELEMENTWISE = {
    "abs",
    "add",
    "sub",
    "mul",
    "div",
    "neg",
    "pow",
    "square",
    "sqrt",
    "rsqrt",
    "exp",
    "log",
    "sigmoid",
    "silu",
    "silu_backward",
    "gelu",
    "gelu_backward",
    "relu",
    "threshold_backward",
    "tanh",
    "tanh_backward",
    "sigmoid_backward",
    "where",
    "eq",
    "ne",
    "gt",
    "ge",
    "lt",
    "le",
    "logical_not",
    "logical_and",
    "logical_or",
    "logical_xor",
    "bitwise_not",
    "clamp",
    "clamp_min",
    "clamp_max",
    "maximum",
    "minimum",
    "reciprocal",
    "sin",
    "cos",
    "conj",
    "resolve_conj",
    "alias",
    "detach",
    "clone",
    "contiguous",
    "_to_copy",
    "convert_element_type",
    "zeros_like",
    "ones_like",
    "empty_like",
}
_MOVING = {"clone", "contiguous", "_to_copy", "convert_element_type", "alias", "detach"}
_RESHAPE = {"view", "reshape", "_unsafe_view", "flatten", "squeeze", "unsqueeze"}


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(value.shape) if isinstance(value, torch.Tensor) else ()


def _same_axis(left: ChunkAxis, right: ChunkAxis, shape: tuple[int, ...]) -> bool:
    # Singleton dimensions can put the same logical factor on different
    # physical axes. Compare ordered coordinates, not the axis spelling.
    def coordinates(axis: ChunkAxis) -> tuple[int, int]:
        return (
            math.prod(shape[: axis.dim]) * axis.before,
            math.prod(shape[axis.dim + 1 :]) * axis.after,
        )

    return coordinates(left) == coordinates(right)


def infer(
    node: Node,
    layouts: Mapping[Node, Any],
    rows: int,
    rules: Mapping[str, OperatorRule],
) -> Any:
    """Infer logical row placement, refusing operations with unproved semantics."""
    if str(node.target) in rules or node.target is operator.getitem:
        return _infer(node, layouts, rows, rules)
    incoming = tree_flatten(
        tree_map(lambda x: layouts[x] if isinstance(x, Node) else None, node.args)
    )[0]
    if not any(isinstance(x, LocalLayout) for x in incoming):
        return _infer(node, layouts, rows, rules)
    if any(isinstance(x, ChunkAxis) for x in incoming):
        raise CaptureError(
            "chunk-local coordinates need an explicit row-restoration rule"
        )
    normalized = {
        key: tree_map(lambda x: x.axis if isinstance(x, LocalLayout) else x, value)
        for key, value in layouts.items()
    }
    result = _infer(node, normalized, rows, rules)
    return tree_map(
        lambda layout, value: (
            LocalLayout(layout)
            if isinstance(value, torch.Tensor) and layout != REDUCED
            else layout
        ),
        result,
        node.meta.get("val"),
    )


def _infer(
    node: Node,
    layouts: Mapping[Node, Any],
    rows: int,
    rules: Mapping[str, OperatorRule],
) -> Any:
    arguments: tuple[Any, ...] = node.args
    keywords: dict[str, Any] = dict(node.kwargs)
    if any(isinstance(x, Node) for x in tree_flatten(keywords)[0]):
        raise CaptureError(
            "axis contracts require tensor operands as positional arguments"
        )
    args = tree_map(lambda x: layouts[x] if isinstance(x, Node) else None, arguments)
    values = tree_map(
        lambda x: x.meta.get("val") if isinstance(x, Node) else x, arguments
    )
    result: Any = node.meta.get("val")
    if node.target is operator.getitem:
        return layouts[arguments[0]][arguments[1]]
    target = str(node.target)
    if target in rules:
        rule = rules[target]
        outputs, spec = tree_flatten(result)
        if len(outputs) != len(rule.output_axes):
            raise CaptureError(f"row rule {target} has a different output count")
        mapped: list[Layout] = []
        for i, (value, dim) in enumerate(zip(outputs, rule.output_axes, strict=True)):
            if value is None or (
                isinstance(value, torch.Tensor) and tuple(value.shape) == (0,)
            ):
                mapped.append(None)
            elif isinstance(dim, ChunkLocal):
                axis = None
                if dim.dim is not None:
                    if (
                        not isinstance(value, torch.Tensor)
                        or not -value.ndim <= dim.dim < value.ndim
                        or value.shape[dim.dim] % rows
                    ):
                        raise CaptureError(
                            f"row rule {target} has invalid local geometry"
                        )
                    axis = ChunkAxis(
                        dim.dim % value.ndim, after=value.shape[dim.dim] // rows
                    )
                mapped.append(LocalLayout(axis))
            elif isinstance(dim, ChunkAxis):
                if (
                    not isinstance(value, torch.Tensor)
                    or not -value.ndim <= dim.dim < value.ndim
                    or value.shape[dim.dim] != dim.before * rows * dim.after
                ):
                    raise CaptureError(f"row rule {target} has invalid factored output")
                mapped.append(ChunkAxis(dim.dim % value.ndim, dim.before, dim.after))
            elif isinstance(dim, AxisFrom):
                if dim.input_index >= len(args):
                    raise CaptureError(
                        f"row rule {target} input position is out of range"
                    )
                axis = args[dim.input_index]
                local = isinstance(axis, LocalLayout)
                if local:
                    axis = axis.axis
                shape = _shape(values[dim.input_index])
                if not isinstance(axis, ChunkAxis):
                    mapped.append(LocalLayout(axis) if local else axis)
                    continue
                if dim.trailing_dims:
                    shape = shape[: -dim.trailing_dims]
                if axis.dim >= len(shape):
                    raise CaptureError(f"row rule {target} discards its chunk axis")
                derived = reshape_axis(axis, shape, _shape(value), rows)
                mapped.append(LocalLayout(derived) if local else derived)
            elif dim is not None:
                if (
                    not isinstance(value, torch.Tensor)
                    or not -value.ndim <= dim < value.ndim
                    or value.shape[dim] != rows
                ):
                    raise CaptureError(
                        f"row rule {target} output {i} has no declared chunk axis"
                    )
                mapped.append(ChunkAxis(dim % value.ndim))
            else:
                mapped.append(REDUCED if i in rule.reductions else None)
        from torch.utils._pytree import tree_unflatten

        return tree_unflatten(mapped, spec)
    parts = target.split(".")
    namespace, name = (parts[0], parts[1]) if len(parts) > 1 else ("", target)
    if namespace == "shadowspill" and name == "materialize_gradient":
        # This allocation changes storage ownership, not coordinates or values.
        # Preserve reductions as well as sliced and shared gradient layouts.
        return args[0]
    flat = tree_flatten(args)[0]
    axes = [x for x in flat if isinstance(x, ChunkAxis)]
    reduced = REDUCED in flat
    # These factories read the source's dtype/device, not its row values.
    # Literal geometry stays unchanged; other consumers must align it.
    if (
        namespace == "aten"
        and name in {"new_zeros", "new_ones", "new_full"}
        and not any(isinstance(x, Node) for x in tree_flatten(arguments[1:])[0])
    ):
        return None
    if namespace == "shadowspill" and name == "accumulate_matmul_":
        contribution = _matmul_layout(
            args[1], args[2], values[1], values[2], _shape(values[0]), target
        )
        expected = args[0] if isinstance(args[0], ChunkAxis) else REDUCED
        if contribution != expected:
            raise CaptureError(
                "gradient matrix accumulation has incompatible chunk domains"
            )
        return None
    if namespace not in {"aten", "prims"}:
        raise CaptureError(
            f"no row-separability rule for {target}; register an OperatorRule"
        )
    if (
        name in {"new_empty", "new_zeros", "empty", "empty_strided"}
        and isinstance(result, torch.Tensor)
        and result.numel() == 0
    ):
        return None
    if reduced and name not in (
        _MOVING
        | _RESHAPE
        | {
            "neg",
            "add",
            "sub",
            "mul",
            "div",
            "add_",
            "mm",
            "bmm",
            "matmul",
            "addmm",
            "sum",
            "mean",
            "permute",
            "transpose",
            "t",
            "expand",
            "expand_as",
            "broadcast_in_dim",
            "slice",
            "select",
            "split",
            "split_with_sizes",
            "unbind",
            "cat",
            "stack",
            "slice_backward",
            "select_backward",
            "linear",
            "_assert_tensor_metadata",
        }
    ):
        raise CaptureError(f"{target} nonlinearly transforms a row reduction")
    if name == "_assert_tensor_metadata":
        size = arguments[1] if len(arguments) > 1 else keywords.get("size")
        stride = arguments[2] if len(arguments) > 2 else keywords.get("stride")
        if size is not None or stride is not None:
            raise CaptureError(
                "axis chunking cannot preserve fixed size/stride assertions"
            )
        return None
    if name in {"add_", "copy_"}:
        if name == "copy_":
            raise CaptureError("chunked tasks require additive gradient mutations")
        if isinstance(args[1], ChunkAxis) and args[0] != args[1]:
            raise CaptureError("an additive mutation must align its chunk domains")
        if args[0] is None and args[1] != REDUCED:
            raise CaptureError("a shared gradient mutation must add a row reduction")
        return args[0]
    if name in _ELEMENTWISE:
        if reduced:
            allowed = name in _MOVING | {"neg", "add", "sub", "mul", "div"}
            tensor_args = [x for x in args if x is not None]
            if not allowed or axes:
                raise CaptureError(
                    f"{target} mixes or nonlinearly transforms a row reduction"
                )
            if name in {"add", "sub"} and any(
                a != REDUCED
                for a, v in zip(args, values, strict=True)
                if isinstance(v, torch.Tensor)
                or (isinstance(v, (float, int)) and v != 0)
            ):
                raise CaptureError(f"{target} adds a shared value to a row reduction")
            if name == "div" and args[0] != REDUCED:
                raise CaptureError(f"{target} divides by a row reduction")
            if name in {"mul", "div"} and len(tensor_args) > 1:
                raise CaptureError(f"{target} multiplies two chunk reductions")
            return REDUCED
        mapped_axes = []
        for a, v in zip(args, values, strict=True):
            if isinstance(a, ChunkAxis):
                mapped_axes.append(
                    ChunkAxis(
                        a.dim + len(_shape(result)) - len(_shape(v)), a.before, a.after
                    )
                )
        if mapped_axes and any(
            not _same_axis(a, mapped_axes[0], _shape(result)) for a in mapped_axes
        ):
            raise CaptureError(f"{target} combines differently aligned chunk domains")
        return mapped_axes[0] if mapped_axes else None
    if name in _RESHAPE:
        return (
            reshape_axis(args[0], _shape(values[0]), _shape(result), rows)
            if isinstance(args[0], ChunkAxis)
            else args[0]
        )
    if name in {"permute", "transpose", "t"}:
        axis = args[0]
        if not isinstance(axis, ChunkAxis):
            return axis
        rank = len(_shape(values[0]))
        if name == "permute":
            order = tuple(d % rank for d in arguments[1])
        elif name == "t":
            order = tuple(reversed(range(rank)))
        else:
            order_list = list(range(rank))
            a, b = int(arguments[1]) % rank, int(arguments[2]) % rank
            order_list[a], order_list[b] = order_list[b], order_list[a]
            order = tuple(order_list)
        return ChunkAxis(order.index(axis.dim), axis.before, axis.after)
    if name in {"sum", "mean", "amax", "amin", "var", "var_mean"}:
        axis = args[0]
        if not isinstance(axis, ChunkAxis):
            return axis
        rank = len(_shape(values[0]))
        dims = arguments[1] if len(arguments) > 1 else list(range(rank))
        if isinstance(dims, int):
            dims = [dims]
        dims = [d % rank for d in (dims or list(range(rank)))]
        if axis.dim in dims:
            if name != "sum":
                raise CaptureError(f"{target} reduces the chunk axis non-additively")
            return REDUCED
        keepdim = (
            bool(arguments[2])
            if len(arguments) > 2
            else bool(keywords.get("keepdim", False))
        )
        dim = axis.dim if keepdim else axis.dim - sum(d < axis.dim for d in dims)
        answer = ChunkAxis(dim, axis.before, axis.after)
        return tuple(answer for _ in result) if isinstance(result, tuple) else answer
    if name in {"mm", "bmm", "matmul", "addmm"}:
        offset = 1 if name == "addmm" else 0
        output = _matmul_layout(
            args[offset],
            args[offset + 1],
            values[offset],
            values[offset + 1],
            _shape(result),
            target,
        )
        if name == "addmm" and keywords.get("beta", 1) != 0:
            bias = args[0]
            if (output == REDUCED) != (bias == REDUCED):
                raise CaptureError(f"{target} adds a shared value to a row reduction")
            if isinstance(bias, ChunkAxis) and bias != output:
                raise CaptureError(f"{target} bias has an incompatible chunk domain")
        return output
    if name == "linear":
        axis = args[0]
        if any(x is not None for x in args[1:]):
            raise CaptureError("linear axis chunking requires shared weight and bias")
        if isinstance(axis, ChunkAxis) and axis.dim == len(_shape(values[0])) - 1:
            raise CaptureError("linear contracts the logical chunk axis")
        if axis == REDUCED and len(values) > 2 and values[2] is not None:
            raise CaptureError("linear adds a shared bias to a row reduction")
        return axis
    if name in {"expand", "expand_as", "broadcast_in_dim"}:
        axis = args[0]
        if isinstance(axis, ChunkAxis):
            return ChunkAxis(
                axis.dim + len(_shape(result)) - len(_shape(values[0])),
                axis.before,
                axis.after,
            )
        return axis
    if name in {"slice_backward", "select_backward"}:
        axis = args[0]
        if not isinstance(axis, ChunkAxis):
            return axis
        rank = len(_shape(result))
        dim = int(arguments[2]) % rank
        if name == "select_backward" and dim <= axis.dim:
            axis = ChunkAxis(axis.dim + 1, axis.before, axis.after)
        if axis.dim == dim:
            raise CaptureError(f"{target} writes part of the logical chunk axis")
        return axis
    if name == "stack":
        incoming = args[0]
        if len(set(incoming)) != 1:
            raise CaptureError("stack inputs carry different row/reduction domains")
        axis = incoming[0]
        if isinstance(axis, ChunkAxis):
            dim = (int(arguments[1]) if len(arguments) > 1 else 0) % len(_shape(result))
            return ChunkAxis(axis.dim + (dim <= axis.dim), axis.before, axis.after)
        return axis
    if name in {"slice", "select", "split", "split_with_sizes", "unbind", "cat"}:
        if name == "cat":
            incoming = args[0]
            if len(set(incoming)) != 1:
                raise CaptureError("cat inputs carry different chunk domains")
            axis = incoming[0]
            dim = int(arguments[1]) if len(arguments) > 1 else 0
            rank = len(_shape(result))
        else:
            axis = args[0]
            dim_position = 2 if name in {"split", "split_with_sizes"} else 1
            dim = int(arguments[dim_position]) if len(arguments) > dim_position else 0
            rank = len(_shape(values[0]))
        dim %= rank
        if isinstance(axis, ChunkAxis) and dim == axis.dim:
            identity_slice = (
                name == "slice"
                and (len(arguments) < 3 or arguments[2] == 0)
                and (len(arguments) < 4 or arguments[3] >= _shape(values[0])[dim])
                and (len(arguments) < 5 or arguments[4] == 1)
            )
            if not identity_slice:
                raise CaptureError(
                    f"{target} slices or assembles the logical chunk axis"
                )
        if (
            isinstance(axis, ChunkAxis)
            and name in {"select", "unbind"}
            and dim < axis.dim
        ):
            axis = ChunkAxis(axis.dim - 1, axis.before, axis.after)
        return tree_map(lambda _: axis, result)
    if name in {
        "_softmax",
        "softmax",
        "_log_softmax",
        "log_softmax",
        "_softmax_backward_data",
        "_log_softmax_backward_data",
    }:
        axis = axes[0] if axes else None
        if axis is not None and any(
            not _same_axis(other, axis, _shape(result)) for other in axes
        ):
            raise CaptureError(f"{target} combines differently aligned chunk domains")
        dim = int(arguments[2] if "backward" in name else arguments[1]) % len(
            _shape(result)
        )
        if axis and axis.dim == dim:
            raise CaptureError(f"{target} mixes rows through softmax")
        return axis
    if name in {"layer_norm", "native_layer_norm", "native_layer_norm_backward"}:
        backward = name == "native_layer_norm_backward"
        source = 1 if backward else 0
        parameters = (5, 6) if backward else (2, 3)
        if any(args[i] is not None for i in parameters if i < len(args)):
            raise CaptureError("layer normalization requires shared weight and bias")
        axis = args[source]
        if not isinstance(axis, ChunkAxis):
            if axes:
                raise CaptureError("layer normalization has incompatible chunk domains")
            return tree_map(lambda _: None, result)
        shape = _shape(values[source])
        if axis.dim >= len(shape) - len(arguments[source + 1]):
            raise CaptureError("layer normalization includes the chunk domain")
        if backward:
            # Equal extents do not establish equal coordinates: the cotangent,
            # activation and saved statistics must slice the same coordinates.
            for i in (0, 3, 4):
                other = args[i]
                if (
                    not isinstance(other, ChunkAxis)
                    or len(_shape(values[i])) != len(shape)
                    or not _same_axis(other, axis, shape)
                ):
                    raise CaptureError(
                        "layer normalization combines differently aligned chunk domains"
                    )
            return tuple(
                layout if isinstance(value, torch.Tensor) else None
                for layout, value in zip((axis, REDUCED, REDUCED), result, strict=True)
            )
        return tree_map(lambda _: axis, result)
    if name == "embedding":
        if args[0] is not None:
            raise CaptureError("embedding axis chunking requires a shared table")
        return args[1]
    if name == "index" and args[0] is None and len(args[1]) == 1:
        # A shared table indexed by row-local positions (for example position
        # encodings) preserves the index tensor's ordered chunk domain.
        axis = args[1][0]
        if isinstance(axis, ChunkAxis):
            return axis
    if name == "embedding_dense_backward":
        if axes and arguments[4]:
            raise CaptureError(
                "embedding gradient frequency scaling needs complete-axis extents"
            )
        if axes and args[0] != args[1]:
            raise CaptureError("embedding gradients and indices must align rows")
        return REDUCED if axes else None
    if (
        not axes
        and not reduced
        and name
        in {
            "scalar_tensor",
            "arange",
            "full",
            "zeros",
            "ones",
            "empty",
            "empty_strided",
            "lift_fresh_copy",
        }
    ):
        return None
    raise CaptureError(
        f"no row-separability rule for {target}; split this region or register a rule"
    )


def _matmul_layout(
    left: Layout,
    right: Layout,
    lv: Any,
    rv: Any,
    result_shape: tuple[int, ...],
    target: str,
) -> Layout:
    axes = [x for x in (left, right) if isinstance(x, ChunkAxis)]
    lr, rr = len(_shape(lv)), len(_shape(rv))
    if lr < 2 or rr < 2:
        raise CaptureError(
            f"{target} needs explicit matrix dimensions for axis chunking"
        )
    if left == REDUCED or right == REDUCED:
        if axes or left == right:
            raise CaptureError(f"{target} mixes chunk reductions")
        return REDUCED
    lc = isinstance(left, ChunkAxis) and left.dim == lr - 1
    rc = isinstance(right, ChunkAxis) and right.dim == rr - 2
    if lc or rc:
        if (
            not (lc and rc)
            or not isinstance(left, ChunkAxis)
            or not isinstance(right, ChunkAxis)
            or left.before != right.before
            or left.after != right.after
        ):
            raise CaptureError(
                f"{target} contracts a chunk domain with an unsliced value"
            )
        return REDUCED
    outputs = []
    if isinstance(left, ChunkAxis):
        dim = left.dim + len(result_shape) - lr
        outputs.append(ChunkAxis(dim, left.before, left.after))
    if isinstance(right, ChunkAxis):
        dim = right.dim + len(result_shape) - rr
        outputs.append(ChunkAxis(dim, right.before, right.after))
    if outputs and any(not _same_axis(a, outputs[0], result_shape) for a in outputs):
        raise CaptureError(f"{target} forms a cross-row product")
    return outputs[0] if outputs else None
