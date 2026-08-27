"""A backward can be rewritten to add its gradients onto given ones.

Every microbatch after the first contributes to a gradient that already
exists. Without this the graph returns a fresh gradient and something outside
it performs the addition, which puts device work between tasks that no plan
accounts for.
"""

from __future__ import annotations

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.aot import (
    accumulate_gradient_outputs,
    capture_graph_pair,
)
from shadowspill.pytorch.capture.artifacts import GraphArtifact, TaskInputRole


def _two_layer_backward(device: str) -> tuple[GraphArtifact, tuple[int, ...]]:
    """Capture one backward, and say which of its outputs are gradients."""

    first = torch.randn(64, 32, device=device, requires_grad=True)
    second = torch.randn(16, 64, device=device, requires_grad=True)
    values = torch.randn(8, 32, device=device)

    def forward(one: torch.Tensor, two: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x @ one.t()) @ two.t()

    graph = make_fx(forward)(first, second, values)
    pair = capture_graph_pair(
        graph,
        (first, second, values),
        original_output=forward(first, second, values),
        recomputation=False,
    )
    output = next(
        node for node in pair.backward.graph_module.graph.nodes if node.op == "output"
    )
    leaves = tuple(
        index for index, value in enumerate(output.args[0]) if value is not None
    )
    return pair.backward, leaves


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
