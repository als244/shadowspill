"""Make selected native scalar rounding explicit before AOT decomposition."""

from __future__ import annotations

from typing import Any

import torch
from torch.fx import GraphModule, Node
from torch.fx.passes.shape_prop import _extract_tensor_metadata

from shadowspill.pytorch.capture.torch_deprecations import copy_graph_module

_SIGMOID_BACKWARD = torch.ops.aten.sigmoid_backward.default
_CUDA_SIGMOID_SCALAR = "cuda.sigmoid_backward.scalar_rounding.v1"


def native_pointwise_rounding(
    graph_module: GraphModule, *, emulate: bool
) -> tuple[GraphModule, tuple[str, ...]]:
    """Preserve CUDA half/bfloat16 sigmoid backward's native scalar arithmetic.

    Native CUDA rounds the complement and the first product at the operands'
    dtype. AOT's default decomposition uses float32 operator math internally;
    marking its final output alone cannot reproduce those intermediate rounds.
    Explicit casts preserve them inside a fused pointwise kernel. This pass
    does not change CPU, other dtypes, or the compiler's non-emulating path.
    """
    if not emulate or torch.version.hip is not None:
        return graph_module, ()
    if not any(_eligible(node) for node in graph_module.graph.nodes):
        return graph_module, ()
    rewritten = copy_graph_module(graph_module)
    graph = rewritten.graph
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
            like: torch.Tensor,
            result_dtype: torch.dtype,
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
            complement = operation(
                torch.ops.aten.rsub.Scalar,
                output_float,
                1.0,
                like=output.meta["val"],
                result_dtype=torch.float32,
            )
            complement_float = convert(convert(complement, dtype), torch.float32)
            product = operation(
                torch.ops.aten.mul.Tensor,
                grad_float,
                complement_float,
                like=value,
                result_dtype=torch.float32,
            )
            product_float = convert(convert(product, dtype), torch.float32)
            final = operation(
                torch.ops.aten.mul.Tensor,
                product_float,
                output_float,
                like=value,
                result_dtype=torch.float32,
            )
            replacement = convert(final, dtype)
        replacement.meta.update(node.meta)
        node.replace_all_uses_with(replacement)
        graph.erase_node(node)
    graph.lint()
    rewritten.recompile()
    return rewritten, (_CUDA_SIGMOID_SCALAR,)


def _eligible(node: Node) -> bool:
    if node.op != "call_function" or node.target != _SIGMOID_BACKWARD:
        return False
    value = node.meta.get("val")
    if (
        not isinstance(value, torch.Tensor)
        or value.layout is not torch.strided
        or value.device.type != "cuda"
        or value.dtype not in (torch.bfloat16, torch.float16)
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
