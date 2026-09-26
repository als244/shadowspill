"""A backward can be rewritten to add its gradients onto given ones, and to
give them at a dtype of their own.

Every microbatch after the first contributes to a gradient that already
exists. Without this the graph returns a fresh gradient and something outside
it performs the addition, which puts device work between tasks that no plan
accounts for. A gradient kept at another dtype than its parameter's -- fp32
for bf16 weights, say -- comes out of the backward at that dtype, so it is
added there too.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.aot import (
    accumulate_gradient_outputs,
    capture_graph_pair,
    cast_gradient_outputs,
)
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.task.inputs import TaskInputRole


def _backward(
    forward: Callable[..., torch.Tensor], inputs: tuple[torch.Tensor, ...]
) -> tuple[GraphArtifact, tuple[int, ...]]:
    """Capture one backward, and say which of its outputs are gradients."""

    graph = make_fx(forward)(*inputs)
    pair = capture_graph_pair(
        graph, inputs, original_output=forward(*inputs), recomputation=False
    )
    leaves = tuple(
        index
        for index, value in enumerate(_output(pair.backward).args[0])
        if value is not None
    )
    return pair.backward, leaves


def _output(artifact: GraphArtifact) -> torch.fx.Node:
    return next(
        node for node in artifact.graph_module.graph.nodes if node.op == "output"
    )


def _two_layer_backward(device: str) -> tuple[GraphArtifact, tuple[int, ...]]:
    first = torch.randn(64, 32, device=device, requires_grad=True)
    second = torch.randn(16, 64, device=device, requires_grad=True)
    values = torch.randn(8, 32, device=device)

    def forward(one: torch.Tensor, two: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x @ one.t()) @ two.t()

    return _backward(forward, (first, second, values))


def _projection_backward(
    device: str, dtype: torch.dtype = torch.bfloat16
) -> tuple[GraphArtifact, tuple[int, ...]]:
    """``x @ weight.t() + bias``: the weight's gradient is a matrix multiply of
    the backward's inputs, moved by a transpose on its way out; the bias's is
    a sum."""

    weight = torch.randn(16, 32, dtype=dtype, device=device, requires_grad=True)
    bias = torch.randn(16, dtype=dtype, device=device, requires_grad=True)
    values = torch.randn(64, 32, dtype=dtype, device=device)

    def forward(w: torch.Tensor, b: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x @ w.t() + b

    return _backward(forward, (weight, bias, values))


def _materialize(value: object, device: str) -> object:
    """Example arguments are fake, so make something runnable of that shape."""

    if isinstance(value, torch.Tensor):
        return torch.randn(tuple(value.shape), dtype=value.dtype, device=device)
    return value


@pytest.mark.cuda
def test_rewritten_backward_adds_into_the_gradient_it_is_given() -> None:
    """The sum lands in the storage that was passed in, not in a new tensor."""

    backward, leaves = _two_layer_backward("cuda")
    accumulating = accumulate_gradient_outputs(backward, leaves)

    arguments = tuple(_materialize(item, "cuda") for item in backward.example_arguments)
    plain = backward.graph_module(*arguments)
    priors = [torch.randn_like(plain[leaf]) for leaf in leaves]
    originals = [prior.clone() for prior in priors]
    total = accumulating.graph_module(*arguments, *priors)

    for index, leaf in enumerate(leaves):
        expected = plain[leaf] + originals[index]
        assert torch.allclose(total[leaf], expected, rtol=1e-4, atol=1e-4)
        assert total[leaf].data_ptr() == priors[index].data_ptr()


@pytest.mark.cuda
def test_outputs_that_are_not_gradients_are_left_alone() -> None:
    backward, leaves = _two_layer_backward("cuda")
    accumulating = accumulate_gradient_outputs(backward, leaves[:1])

    arguments = tuple(_materialize(item, "cuda") for item in backward.example_arguments)
    plain = backward.graph_module(*arguments)
    prior = torch.randn_like(plain[leaves[0]])
    total = accumulating.graph_module(*arguments, prior)

    for index, value in enumerate(plain):
        if index == leaves[0] or not isinstance(value, torch.Tensor):
            continue
        assert torch.allclose(total[index], value)


@pytest.mark.cuda
def test_the_rewrite_declares_what_it_accumulates_onto() -> None:
    """The runtime learns from this that the result replaces the argument."""

    backward, leaves = _two_layer_backward("cuda")
    accumulating = accumulate_gradient_outputs(backward, leaves)

    added = len(accumulating.example_arguments) - len(backward.example_arguments)
    assert added == len(leaves)
    assert accumulating.compatibility_digest != backward.compatibility_digest
    assert all(
        item.role is TaskInputRole.GRADIENT
        for item in accumulating.input_provenance[-len(leaves) :]
    )


@pytest.mark.cuda
def test_accumulating_onto_nothing_returns_the_same_backward() -> None:
    backward, _ = _two_layer_backward("cuda")
    assert accumulate_gradient_outputs(backward, ()) is backward


@pytest.mark.cuda
def test_a_leaf_that_is_not_a_gradient_is_refused() -> None:
    """Inputs that need no gradient leave a hole in the outputs."""

    backward, leaves = _two_layer_backward("cuda")
    absent = next(
        index for index in range(backward.output_count) if index not in leaves
    )
    with pytest.raises(CaptureError, match="no gradient output"):
        accumulate_gradient_outputs(backward, (absent,))


def test_a_multiply_writes_the_gradient_it_computes_at_the_dtype_asked_for() -> None:
    """Where PyTorch has a kernel that sums bf16 products into an fp32 result
    -- CUDA's -- the weight's gradient is that result, never rounded to bf16.
    The bias's gradient, a sum, is cast; either way the running gradient the
    accumulating form adds onto is fp32."""

    with FakeTensorMode():
        backward, leaves = _projection_backward("cuda")
    kept = cast_gradient_outputs(backward, leaves, torch.float32)

    weight, bias = (_output(kept).args[0][leaf] for leaf in leaves)
    targets = {node.target for node in kept.graph_module.graph.nodes}
    assert torch.ops.aten.mm.dtype in targets
    assert torch.ops.aten.mm.default not in targets
    assert weight.target is not torch.ops.prims.convert_element_type.default
    assert bias.target is torch.ops.prims.convert_element_type.default
    assert weight.meta["val"].dtype == bias.meta["val"].dtype == torch.float32
    accumulating = accumulate_gradient_outputs(kept, leaves)
    priors = accumulating.example_arguments[-len(leaves) :]
    assert all(item.dtype == torch.float32 for item in priors)


def test_a_dtype_the_multiply_does_not_write_is_cast() -> None:
    """Which dtypes a multiply writes is the operator's own check: CUDA's
    writes fp32 from bf16 or fp16, not fp64 from fp32, so fp64 gradients of
    fp32 weights are cast."""

    with FakeTensorMode():
        backward, leaves = _projection_backward("cuda", torch.float32)
    kept = cast_gradient_outputs(backward, leaves, torch.float64)

    assert all(
        _output(kept).args[0][leaf].target
        is torch.ops.prims.convert_element_type.default
        for leaf in leaves
    )
    assert torch.ops.aten.mm.default in {
        node.target for node in kept.graph_module.graph.nodes
    }


def test_where_no_kernel_writes_the_dtype_the_gradient_is_cast() -> None:
    """PyTorch has no CPU kernel that writes a multiply at another dtype: on
    the CPU both gradients are the bf16 results, cast."""

    backward, leaves = _projection_backward("cpu")
    kept = cast_gradient_outputs(backward, leaves, torch.float32)

    arguments = tuple(_materialize(item, "cpu") for item in backward.example_arguments)
    plain = backward.graph_module(*arguments)
    cast = kept.graph_module(*arguments)
    for leaf in leaves:
        assert cast[leaf].dtype == torch.float32
        assert torch.equal(cast[leaf], plain[leaf].float())


@pytest.mark.cuda
def test_the_multiplys_gradient_is_summed_at_fp32() -> None:
    """Within fp32's error of the exact product of the same bf16 operands,
    where the bf16 result is off by bf16's."""

    backward, leaves = _projection_backward("cuda")
    kept = cast_gradient_outputs(backward, leaves, torch.float32)

    arguments = tuple(_materialize(item, "cuda") for item in backward.example_arguments)
    wide = (
        item.double() if isinstance(item, torch.Tensor) else item for item in arguments
    )
    exact = backward.graph_module(*wide)[leaves[0]]
    written = kept.graph_module(*arguments)[leaves[0]]
    rounded = backward.graph_module(*arguments)[leaves[0]]
    assert written.dtype == torch.float32
    error = (written.double() - exact).abs().max()
    assert error <= 1e-5 * exact.abs().max()
    assert error * 100 < (rounded.double() - exact).abs().max()


@pytest.mark.cuda
def test_the_accumulating_form_adds_at_the_gradient_dtype() -> None:
    backward, leaves = _projection_backward("cuda")
    kept = cast_gradient_outputs(backward, leaves, torch.float32)
    accumulating = accumulate_gradient_outputs(kept, leaves)

    arguments = tuple(_materialize(item, "cuda") for item in backward.example_arguments)
    contributions = kept.graph_module(*arguments)
    priors = [
        torch.randn_like(contributions[leaf], dtype=torch.float32) for leaf in leaves
    ]
    originals = [prior.clone() for prior in priors]
    total = accumulating.graph_module(*arguments, *priors)

    for index, leaf in enumerate(leaves):
        assert total[leaf].data_ptr() == priors[index].data_ptr()
        assert torch.equal(total[leaf], originals[index] + contributions[leaf])
