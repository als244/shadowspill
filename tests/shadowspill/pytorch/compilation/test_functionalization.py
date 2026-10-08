"""Direct task lowering must preserve mutations before optimizing lifetimes."""

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.compilation.compiler import compile_artifact


def _compile(function, inputs):
    graph = make_fx(function, tracing_mode="fake")(*inputs)
    artifact = GraphArtifact.capture(
        kind="optimizer", graph_module=graph, example_inputs=inputs
    )
    return compile_artifact(artifact, device_ordinal=0).function


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mutated_temporary_survives_later_scratch_reuse(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    def operation(weight, numerator, denominator, other, moment, gradient):
        owned = torch.nn.functional.pad(weight, (0, 3))[:3].clone()
        updated = owned.addcdiv_(numerator, denominator)
        scratch = -(gradient.abs() + 0.0001) / 0.025
        other_updated = other.clone().addcdiv_(moment, scratch)
        return torch.cat((updated, other_updated))

    generator = torch.Generator().manual_seed(42)
    inputs = tuple(
        torch.randn(size, generator=generator).to(device)
        for size in (9, 3, 3, 12, 12, 12)
    )
    inputs[2].abs_().add_(1)
    originals = tuple(value.clone() for value in inputs)
    expected = operation(*inputs)
    compiled = _compile(operation, inputs)
    for _ in range(3):
        torch.testing.assert_close(compiled(*inputs), expected)
        # Clone elimination must never make the internal updates mutate inputs.
        torch.testing.assert_close(inputs, originals, rtol=0, atol=0)


def test_overlapping_input_views_share_the_functional_update():
    def operation(left, right):
        left.add_(1)
        return right.square()

    storage = torch.ones(6)
    inputs = (storage[:4], storage[2:])
    compiled = _compile(operation, inputs)
    actual = compiled(*inputs)
    torch.testing.assert_close(actual, torch.tensor([4.0, 4.0, 1.0, 1.0]))
    torch.testing.assert_close(storage, torch.tensor([2.0, 2.0, 2.0, 2.0, 1.0, 1.0]))


@torch.library.custom_op("shadowspill_compile_test::mutate", mutates_args=("value",))
def _mutate(value: torch.Tensor) -> torch.Tensor:
    value.add_(1)
    return value.square()


@_mutate.register_fake
def _mutate_fake(value):
    return torch.empty_like(value)


def test_mutable_opaque_operator_preserves_input_update():
    value = torch.ones(4)
    compiled = _compile(_mutate, (value,))
    torch.testing.assert_close(compiled(value), torch.full((4,), 4.0))
    torch.testing.assert_close(value, torch.full((4,), 2.0))
