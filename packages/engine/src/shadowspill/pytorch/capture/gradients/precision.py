"""Preserve requested gradient precision through casts and GEMM outputs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import nullcontext
from typing import Any, cast

import torch
from torch._guards import detect_fake_mode

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.artifacts import GraphArtifact

from ..torch_deprecations import copy_graph_module
from .nodes import (
    _CONVERTING,
    _MOVING_ONLY,
    _WRITING_AT_DTYPE,
    _record_value,
)


def cast_gradient_outputs(
    backward: GraphArtifact,
    leaf_indices: Sequence[int],
    dtype: torch.dtype,
) -> GraphArtifact:
    """Return a backward whose gradients at ``leaf_indices`` come out at ``dtype``.

    A backward computes a gradient at the dtype of what it is the gradient of.
    One kept at another dtype -- fp32 gradients of bf16 weights, say -- comes
    out at that one: a gradient the step creates is created at it, and the
    accumulating form, derived from this one, adds each contribution into the
    running gradient there.

    Which operation computed a gradient decides how. One an operation
    returned at ``dtype`` already -- a kernel asked for fp32 weight gradients
    -- reaches the output converted to the parameter's dtype, since autograd
    gives a parameter its gradient at the parameter's dtype; the conversion is
    dropped and the value the operation returned kept. One a matrix multiply
    computes, moved at most by views and copies on its way out, is written at
    ``dtype`` by the multiply itself where PyTorch has a kernel for it (fp32
    from fp16 or bf16 operands on CUDA, say): its products are summed at
    ``dtype`` and never rounded to the operands'. Any other is what the
    operations computed, cast as it leaves.
    """

    graph_module = copy_graph_module(backward.graph_module)
    graph = graph_module.graph
    output_node = next(node for node in graph.nodes if node.op == "output")
    outputs = list(output_node.args[0])
    fake_mode = detect_fake_mode(backward.example_arguments)
    changed = False
    for leaf in leaf_indices:
        produced = outputs[leaf] if 0 <= leaf < len(outputs) else None
        if produced is None or not isinstance(produced.meta.get("val"), torch.Tensor):
            raise CaptureError(f"backward gradient at leaf {leaf} has no geometry")
        value = produced.meta["val"]
        if value.dtype == dtype:
            continue
        changed = True
        unconverted = _before_conversion(graph, produced, outputs, dtype, fake_mode)
        if unconverted is not None:
            outputs[leaf] = unconverted
            continue
        written = _write_at_dtype(graph, produced, outputs, dtype, fake_mode)
        if written is not None:
            outputs[leaf] = written
            continue
        with graph.inserting_before(output_node):
            cast = graph.call_function(
                torch.ops.prims.convert_element_type.default, args=(produced, dtype)
            )
        cast.meta = dict(produced.meta)
        with fake_mode if fake_mode is not None else nullcontext():
            _record_value(cast, value.to(dtype))
        outputs[leaf] = cast
    if not changed:
        return backward
    output_node.args = (tuple(outputs),)
    graph.lint()
    graph_module.recompile()
    return GraphArtifact.capture(
        kind="backward",
        graph_module=graph_module,
        example_inputs=backward.example_arguments,
        input_provenance=backward.input_provenance,
    )


def _before_conversion(
    graph: torch.fx.Graph,
    produced: torch.fx.Node,
    outputs: Sequence[object],
    dtype: torch.dtype,
    fake_mode: Any,
) -> torch.fx.Node | None:
    """The gradient ``produced`` is converted from, where that is at ``dtype``.

    It is when ``produced`` is a dtype conversion, moved at most, that nothing
    else reads, of a value at ``dtype`` already: the moves are applied to that
    value instead, and the conversion goes. Returns the node that now gives the
    gradient, or ``None``.
    """

    chain = [produced]
    while chain[-1].target in _MOVING_ONLY:
        source = chain[-1].args[0]
        if not isinstance(source, torch.fx.Node):
            return None
        chain.append(source)
    conversion = chain[-1]
    source = conversion.args[0] if conversion.args else None
    if (
        conversion.target not in _CONVERTING
        or set(conversion.kwargs) - {"dtype"}
        or not isinstance(source, torch.fx.Node)
        or not isinstance(source.meta.get("val"), torch.Tensor)
        or source.meta["val"].dtype != dtype
        or outputs.count(produced) != 1
        or any(len(node.users) != 1 for node in chain)
    ):
        return None
    conversion.replace_all_uses_with(source)
    graph.erase_node(conversion)
    with fake_mode if fake_mode is not None else nullcontext():
        for node in reversed(chain[:-1]):
            args, kwargs = torch.fx.node.map_arg(
                (node.args, node.kwargs), lambda item: item.meta["val"]
            )
            _record_value(node, cast(Callable[..., Any], node.target)(*args, **kwargs))
    return chain[0] if len(chain) > 1 else source


def _write_at_dtype(
    graph: torch.fx.Graph,
    produced: torch.fx.Node,
    outputs: Sequence[object],
    dtype: torch.dtype,
    fake_mode: Any,
) -> torch.fx.Node | None:
    """Have the matrix multiply that computes ``produced`` write it at ``dtype``.

    It does when ``produced`` is a multiply's result, moved at most, that
    nothing else reads, and PyTorch has a kernel that writes that multiply at
    ``dtype`` from these operands on their device -- which dtypes it accepts
    is the operator's own check. Every node from the multiply to the output
    then carries ``dtype``. Returns the replacement output, or ``None``.
    """

    chain = [produced]
    while chain[-1].target in _MOVING_ONLY:
        source = chain[-1].args[0]
        if not isinstance(source, torch.fx.Node):
            return None
        chain.append(source)
    multiply = chain[-1]
    writing = _WRITING_AT_DTYPE.get(multiply.target)
    if (
        writing is None
        or outputs.count(produced) != 1
        or any(len(node.users) != 1 for node in chain)
    ):
        return None
    operands = torch.fx.node.map_arg(multiply.args, lambda item: item.meta["val"])
    device = cast(torch.Tensor, operands[0]).device
    schema = writing._schema
    if not torch._C._dispatch_has_kernel_for_dispatch_key(
        f"{schema.name}.{schema.overload_name}",
        torch._C._dispatch_key_for_device(device.type),
    ):
        return None
    with fake_mode if fake_mode is not None else nullcontext():
        try:
            value = writing(*operands, dtype, **multiply.kwargs)
        except RuntimeError:  # not a dtype it writes from these operands
            return None
        with graph.inserting_after(multiply):
            written = graph.call_function(
                writing, args=(*multiply.args, dtype), kwargs=dict(multiply.kwargs)
            )
        written.meta = dict(multiply.meta)
        _record_value(written, value)
        multiply.replace_all_uses_with(written)
        graph.erase_node(multiply)
        for node in reversed(chain[:-1]):
            args, kwargs = torch.fx.node.map_arg(
                (node.args, node.kwargs), lambda item: item.meta["val"]
            )
            _record_value(node, cast(Callable[..., Any], node.target)(*args, **kwargs))
    return chain[0] if len(chain) > 1 else written
