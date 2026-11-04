"""Make selected native scalar rounding explicit before AOT decomposition."""

from __future__ import annotations

from typing import Any

import torch
from torch.fx import GraphModule, Node
from torch.fx.passes.shape_prop import _extract_tensor_metadata

from shadowspill.pytorch.capture.torch_deprecations import copy_graph_module

_SIGMOID_BACKWARD = torch.ops.aten.sigmoid_backward.default
_CUDA_SIGMOID_SCALAR = "cuda.sigmoid_backward.scalar_rounding.v1"
_SILU_BACKWARD = torch.ops.aten.silu_backward.default
_CUDA_SILU_OPMATH = "cuda.silu_backward.opmath.v1"


def native_pointwise_rounding(
    graph_module: GraphModule, *, emulate: bool
) -> tuple[GraphModule, tuple[str, ...]]:
    """Preserve selected native CUDA activation derivatives inside fused kernels.

    Native CUDA sigmoid backward rounds the complement and the first product
    at the operands' dtype. AOT decomposition uses float32 operator math internally;
    marking its final output alone cannot reproduce those intermediate rounds.
    Explicit casts preserve them inside a fused pointwise kernel. SiLU uses
    opmath precision, correctly rounded division, and a fused multiply-add;
    portable addcdiv/addcmul express those operations without a custom kernel.
    This pass does not change CPU or the compiler's non-emulating path.
    """
    if not emulate or torch.version.hip is not None:
        return graph_module, ()
    if not any(_eligible(node) for node in graph_module.graph.nodes):
        return graph_module, ()
    rewritten = copy_graph_module(graph_module)
    graph = rewritten.graph
    contracts: set[str] = set()
    for node in tuple(graph.nodes):
        if not _eligible(node):
            continue
        gradient, output = node.args
        assert isinstance(gradient, Node) and isinstance(output, Node)
        value = node.meta["val"]
        dtype = value.dtype

        def operation(
            target: Any,
            *args: Any,
            like: torch.Tensor = value,
            result_dtype: torch.dtype = torch.float32,
            origin: Node = node,
            **kwargs: Any,
        ) -> Node:
            created = graph.call_function(target, args, kwargs)
            created.meta = dict(origin.meta)
            # Captured nodes can belong to different fake modes. The validated
            # pointwise geometry comes from the corresponding original result;
            # do not evaluate a binary operation across unrelated fake modes.
            created.meta["val"] = like.to(dtype=result_dtype)
            created.meta["tensor_meta"] = _extract_tensor_metadata(created.meta["val"])
            return created

        def convert(source: Node, target_dtype: torch.dtype) -> Node:
            return operation(
                torch.ops.aten._to_copy.default,
                source,
                like=source.meta["val"],
                result_dtype=target_dtype,
                dtype=target_dtype,
            )

        with graph.inserting_before(node):
            grad_float = convert(gradient, torch.float32)
            output_float = convert(output, torch.float32)
            if node.target == _SILU_BACKWARD:
                zero = operation(
                    torch.ops.aten.full_like.default,
                    output_float,
                    0.0,
                    dtype=torch.float32,
                )
                one = operation(
                    torch.ops.aten.full_like.default,
                    output_float,
                    1.0,
                    dtype=torch.float32,
                )
                negative = operation(torch.ops.aten.neg.default, output_float)
                exponential = operation(torch.ops.aten.exp.default, negative)
                denominator = operation(torch.ops.aten.add.Tensor, exponential, 1.0)
                # addcdiv lowers to round-to-nearest division rather than the
                # approximate reciprocal used by the default sigmoid lowering.
                sigmoid = operation(
                    torch.ops.aten.addcdiv.default, zero, one, denominator
                )
                complement = operation(torch.ops.aten.rsub.Scalar, sigmoid, 1.0)
                # Make native CUDA's FMA explicit even when general compiler
                # contraction is disabled by precision emulation.
                factor = operation(
                    torch.ops.aten.addcmul.default, one, output_float, complement
                )
                product = operation(torch.ops.aten.mul.Tensor, grad_float, sigmoid)
                final = operation(torch.ops.aten.mul.Tensor, product, factor)
                contracts.add(_CUDA_SILU_OPMATH)
            else:
                complement = operation(
                    torch.ops.aten.rsub.Scalar,
                    output_float,
                    1.0,
                    like=output.meta["val"],
                )
                complement_float = convert(convert(complement, dtype), torch.float32)
                product = operation(
                    torch.ops.aten.mul.Tensor, grad_float, complement_float
                )
                product_float = convert(convert(product, dtype), torch.float32)
                final = operation(
                    torch.ops.aten.mul.Tensor, product_float, output_float
                )
                contracts.add(_CUDA_SIGMOID_SCALAR)
            replacement = convert(final, dtype)
        replacement.meta.update(node.meta)
        node.replace_all_uses_with(replacement)
        graph.erase_node(node)
    graph.lint()
    rewritten.recompile()
    return rewritten, tuple(sorted(contracts))


def _eligible(node: Node) -> bool:
    if node.op != "call_function" or node.target not in (
        _SIGMOID_BACKWARD,
        _SILU_BACKWARD,
    ):
        return False
    value = node.meta.get("val")
    if (
        not isinstance(value, torch.Tensor)
        or value.layout is not torch.strided
        or value.device.type != "cuda"
        or value.dtype
        not in (
            (torch.bfloat16, torch.float16)
            if node.target == _SIGMOID_BACKWARD
            else (torch.bfloat16, torch.float16, torch.float32)
        )
        or len(node.args) != 2
        or node.kwargs
    ):
        return False
    return all(
        isinstance(argument, Node)
        and isinstance(operand := argument.meta.get("val"), torch.Tensor)
        and operand.dtype == value.dtype
        and operand.layout is torch.strided
        and operand.device == value.device
        and operand.shape == value.shape
        for argument in node.args
    )
