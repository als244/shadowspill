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
from typing import Any

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
from shadowspill.pytorch.compilation.compiler import compile_artifact
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


def _kept(
    backward: GraphArtifact, leaves: tuple[int, ...], dtype: torch.dtype | None
) -> GraphArtifact:
    return backward if dtype is None else cast_gradient_outputs(backward, leaves, dtype)


@pytest.mark.parametrize(
    ("gradient_dtype", "round_once"), [(torch.float32, False), (None, True)]
)
def test_a_multiply_adds_the_gradient_it_computes_into_the_running_one(
    gradient_dtype: torch.dtype | None, round_once: bool
) -> None:
    """On CUDA, where a kernel adds a product into a tensor in place, the
    weight's gradient is added by the multiply that computes it -- the
    running gradient is what the backward returns -- when gradients are kept
    at fp32, the dtype the multiply sums bf16 at, and when kept at the
    weights' bf16 if the sum may be rounded once. The bias's sum is added
    after."""

    with FakeTensorMode():
        backward, leaves = _projection_backward("cuda")
    accumulating = accumulate_gradient_outputs(
        _kept(backward, leaves, gradient_dtype),
        leaves,
        round_accumulation_once=round_once,
    )

    nodes = list(accumulating.graph_module.graph.nodes)
    weight, bias = (_output(accumulating).args[0][leaf] for leaf in leaves)
    assert weight.op == "placeholder"
    assert weight.name.startswith("shadowspill_prior_grad_")
    assert [node.target for node in nodes].count(
        torch.ops.shadowspill.accumulate_matmul_.default
    ) == 1
    assert not {torch.ops.aten.mm.default, torch.ops.aten.mm.dtype} & {
        node.target for node in nodes
    }
    assert bias.target is torch.ops.aten.add_.Tensor


def test_a_bf16_running_gradient_is_added_after_the_multiply_by_default() -> None:
    """Kept at bf16, narrower than the multiply sums at, a gradient added
    inside the multiply would be rounded once where adding after rounds the
    product first; by default the step computes what adding after does."""

    with FakeTensorMode():
        backward, leaves = _projection_backward("cuda")
    accumulating = accumulate_gradient_outputs(backward, leaves)

    targets = {node.target for node in accumulating.graph_module.graph.nodes}
    assert torch.ops.shadowspill.accumulate_matmul_.default not in targets
    assert all(
        _output(accumulating).args[0][leaf].target is torch.ops.aten.add_.Tensor
        for leaf in leaves
    )


def test_where_no_kernel_adds_in_place_the_gradient_is_added_after() -> None:
    backward, leaves = _projection_backward("cpu")
    accumulating = accumulate_gradient_outputs(backward, leaves)

    assert all(
        _output(accumulating).args[0][leaf].target is torch.ops.aten.add_.Tensor
        for leaf in leaves
    )


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("gradient_dtype", "tolerance"), [(None, 2**-8), (torch.float32, 1e-5)]
)
def test_the_multiply_adds_in_place_rounding_once(
    gradient_dtype: torch.dtype | None, tolerance: float
) -> None:
    """The sum of the running gradient and the product is rounded once, as it
    is written: within one bf16 rounding of the exact sum at bf16, and within
    fp32's error at fp32."""

    backward, leaves = _projection_backward("cuda")
    accumulating = accumulate_gradient_outputs(
        _kept(backward, leaves, gradient_dtype),
        leaves,
        round_accumulation_once=True,
    )

    arguments = tuple(_materialize(item, "cuda") for item in backward.example_arguments)
    priors = [
        _materialize(item, "cuda")
        for item in accumulating.example_arguments[len(arguments) :]
    ]
    exact = backward.graph_module(*(item.double() for item in arguments))
    expected = [
        exact[leaf] + prior.double() for leaf, prior in zip(leaves, priors, strict=True)
    ]
    total = accumulating.graph_module(*arguments, *priors)

    weight = leaves[0]
    assert total[weight].data_ptr() == priors[0].data_ptr()
    error = (total[weight].double() - expected[0]).abs().max()
    assert error <= tolerance * expected[0].abs().max()


@pytest.mark.cuda
def test_a_batched_multiply_adds_through_the_moves_it_leaves_by() -> None:
    """A gradient that leaves a batched multiply through a permute is added
    where the permute would have put it."""

    weight = torch.randn(4, 32, 16, device="cuda", requires_grad=True)
    values = torch.randn(4, 64, 32, device="cuda")
    backward, leaves = _backward(
        lambda w, x: torch.bmm(x, w).permute(1, 0, 2), (weight, values)
    )
    accumulating = accumulate_gradient_outputs(backward, leaves)
    assert torch.ops.shadowspill.accumulate_matmul_.default in {
        node.target for node in accumulating.graph_module.graph.nodes
    }

    arguments = tuple(_materialize(item, "cuda") for item in backward.example_arguments)
    contribution = backward.graph_module(*arguments)[leaves[0]]
    prior = torch.randn_like(contribution)
    expected = prior + contribution
    total = accumulating.graph_module(*arguments, prior)[leaves[0]]
    assert total.data_ptr() == prior.data_ptr()
    torch.testing.assert_close(total, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.cuda
def test_the_multiply_adding_in_place_compiles_as_one_call() -> None:
    """Compiled, the running gradient is the multiply's output: the backward
    returns it, updated in place, with no product written anywhere else. (The
    bias's sum, compiled, is not rounded to bf16 before it is added, so only
    the weight's gradient is compared with the graph run eagerly.)"""

    backward, leaves = _projection_backward("cuda")
    kept = cast_gradient_outputs(backward, leaves, torch.float32)
    accumulating = accumulate_gradient_outputs(kept, leaves)
    arguments = tuple(
        _materialize(item, "cuda") for item in accumulating.example_arguments
    )
    expected = accumulating.graph_module(*(item.clone() for item in arguments))

    compiled = compile_artifact(
        accumulating, device_ordinal=0, representative_arguments=arguments
    )
    total = compiled()

    for index, leaf in enumerate(leaves):
        prior = arguments[len(kept.example_arguments) + index]
        assert total[leaf].data_ptr() == prior.data_ptr()
    torch.testing.assert_close(total[leaves[0]], expected[leaves[0]])


@pytest.mark.cuda
def test_by_default_a_bf16_step_accumulates_as_adding_after_does() -> None:
    """The running gradient after the default accumulating backward is, bit
    for bit, the product the creating form computes added to it."""

    backward, leaves = _projection_backward("cuda")
    accumulating = accumulate_gradient_outputs(backward, leaves)

    arguments = tuple(_materialize(item, "cuda") for item in backward.example_arguments)
    contributions = backward.graph_module(*arguments)
    priors = [
        _materialize(item, "cuda")
        for item in accumulating.example_arguments[len(arguments) :]
    ]
    expected = [
        prior + contributions[leaf] for leaf, prior in zip(leaves, priors, strict=True)
    ]
    total = accumulating.graph_module(*arguments, *priors)
    for index, leaf in enumerate(leaves):
        assert torch.equal(total[leaf], expected[index])


@torch.library.custom_op("shadowspill_tests::scale", mutates_args=())
def _scale(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x * weight


@_scale.register_fake
def _scale_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    del weight
    return torch.empty_like(x)


@torch.library.custom_op("shadowspill_tests::scale_vjp", mutates_args=())
def _scale_vjp(
    grad: torch.Tensor, x: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The weight's gradient summed at fp32 and returned so, as a kernel asked
    for fp32 weight gradients returns it."""

    return grad * weight, (grad.float() * x.float()).sum(0)


@_scale_vjp.register_fake
def _scale_vjp_fake(
    grad: torch.Tensor, x: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    del grad
    return torch.empty_like(x), torch.empty_like(weight, dtype=torch.float32)


def _scale_context(ctx: Any, inputs: tuple[torch.Tensor, ...], output: object) -> None:
    del output
    ctx.save_for_backward(*inputs)


def _scale_backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x, weight = ctx.saved_tensors
    return _scale_vjp(grad, x, weight)


_scale.register_autograd(_scale_backward, setup_context=_scale_context)

_CONVERSIONS = {
    torch.ops.aten._to_copy.default,
    torch.ops.prims.convert_element_type.default,
}


def test_a_gradient_an_operation_returns_at_the_dtype_asked_for_is_kept() -> None:
    """Autograd gives a parameter its gradient at the parameter's dtype, so an
    operation that returns an fp32 gradient for a bf16 weight has it converted
    as it leaves. Kept at fp32, the conversion goes: the gradient is the value
    the operation returned, not its bf16 rounding widened again."""

    weight = torch.randn(16, dtype=torch.bfloat16, requires_grad=True)
    values = torch.randn(64, 16, dtype=torch.bfloat16)
    backward, leaves = _backward(lambda w, x: _scale(x, w), (weight, values))
    assert _output(backward).args[0][leaves[0]].target in _CONVERSIONS

    kept = cast_gradient_outputs(backward, leaves, torch.float32)
    assert not _CONVERSIONS & {node.target for node in kept.graph_module.graph.nodes}

    arguments = tuple(_materialize(item, "cpu") for item in backward.example_arguments)
    rounded = backward.graph_module(*arguments)[leaves[0]]
    gradient = kept.graph_module(*arguments)[leaves[0]]
    assert gradient.dtype == torch.float32
    assert torch.equal(gradient.to(torch.bfloat16), rounded)
    assert not torch.equal(gradient, rounded.float())
