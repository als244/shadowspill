"""Ordered custom operators work through the ordinary capture APIs."""

from __future__ import annotations

import io

import pytest
import torch
from torch import nn
from torch._higher_order_ops.effects import _EffectType, _register_effectful_op

from shadowspill.pytorch.capture.aot import (
    capture_forward,
    capture_training,
    inference_artifact,
)
from shadowspill.pytorch.graph_pairs.serialization import CachedAotGraphPair

_EVENTS: list[tuple[str, int]] = []


@torch.library.custom_op("shadowspill_effect_test::product", mutates_args=())
def _product(
    x: torch.Tensor, weight: torch.Tensor, label: int, ordered_backward: bool
) -> torch.Tensor:
    _EVENTS.append(("forward", label))
    return x @ weight


@_product.register_fake
def _product_fake(x, weight, label, ordered_backward):
    return x.new_empty((x.shape[0], weight.shape[1]))


@torch.library.custom_op("shadowspill_effect_test::derivative", mutates_args=())
def _derivative(
    dy: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, label: int
) -> tuple[torch.Tensor, torch.Tensor]:
    _EVENTS.append(("backward", label))
    return dy @ weight.T, x.T @ dy


@_derivative.register_fake
def _derivative_fake(dy, x, weight, label):
    return torch.empty_like(x), torch.empty_like(weight)


def _setup(ctx, inputs, output):
    x, weight, ctx.label, ctx.ordered_backward = inputs
    ctx.save_for_backward(x, weight)


def _backward(ctx, dy):
    x, weight = ctx.saved_tensors
    if ctx.ordered_backward:
        dx, dw = _derivative(dy, x, weight, ctx.label)
    else:
        dx, dw = dy @ weight.T, x.T @ dy
    return dx, dw, None, None


_product.register_autograd(_backward, setup_context=_setup)


@torch.library.custom_op("shadowspill_effect_test::marker", mutates_args=())
def _marker(x: torch.Tensor) -> None:
    _EVENTS.append(("marker", 0))


@_marker.register_fake
def _marker_fake(x):
    return None


@pytest.fixture(autouse=True)
def _ordered_operators():
    handles = [
        _register_effectful_op(op, _EffectType.ORDERED)
        for op in (
            torch.ops.shadowspill_effect_test.product.default,
            torch.ops.shadowspill_effect_test.derivative.default,
            torch.ops.shadowspill_effect_test.marker.default,
        )
    ]
    yield
    for handle in handles:
        handle.destroy()
    _EVENTS.clear()


class _Network(nn.Module):
    def __init__(self, ordered_backward: bool = True):
        super().__init__()
        self.weight = nn.Parameter(
            torch.arange(9, dtype=torch.float32).reshape(3, 3) / 10
        )
        self.ordered_backward = ordered_backward

    def forward(self, x):
        x = _product(x, self.weight, 1, self.ordered_backward)
        return _product(x, self.weight, 2, self.ordered_backward)


@pytest.mark.parametrize("ordered_backward", [False, True])
@pytest.mark.parametrize("variant", ["save", "recompute"])
def test_graph_pair_keeps_effect_order_and_correct_gradients(variant, ordered_backward):
    model = _Network(ordered_backward)
    x = torch.arange(12, dtype=torch.float32).reshape(4, 3).requires_grad_()
    expected = (x @ model.weight @ model.weight).square().sum()
    expected_grads = torch.autograd.grad(expected, (model.weight, x))
    captured = capture_training(model, lambda m, values: m(values).square().sum(), (x,))
    pair = getattr(captured, f"{variant}_pair")
    # Exercise the production artifact store as well as in-memory capture.
    storage = io.BytesIO()
    torch.save(CachedAotGraphPair.capture(pair), storage)
    storage.seek(0)
    pair = torch.load(storage, weights_only=False).restore()
    assert pair.forward.argument_count == 2
    assert pair.backward.argument_count == pair.saved_value_count
    assert all(value.numel() > 0 for value in pair.backward.example_arguments)

    _EVENTS.clear()
    outputs = pair.forward.graph_module(model.weight, x)
    torch.testing.assert_close(outputs[0], expected)
    assert _EVENTS == [("forward", 1), ("forward", 2)]
    _EVENTS.clear()
    actual_grads = pair.backward.graph_module(*outputs[1:])
    for actual, reference in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual, reference)
    expected_events = [("forward", 1), ("forward", 2)] if variant == "recompute" else []
    if ordered_backward:
        expected_events += [("backward", 2), ("backward", 1)]
    assert expected_events == _EVENTS


def test_forward_export_preserves_an_ordered_call_with_no_tensor_result():
    class Network(_Network):
        def forward(self, x):
            _marker(x)
            return super().forward(x)

    model = Network()
    x = torch.ones(4, 3)
    artifact = inference_artifact(capture_forward(model, (x,)))
    assert artifact.argument_count == 2
    assert artifact.output_count == 1
    _EVENTS.clear()
    (output,) = artifact.graph_module(model.weight, x)
    torch.testing.assert_close(output, x @ model.weight @ model.weight)
    assert _EVENTS == [("marker", 0), ("forward", 1), ("forward", 2)]


def test_export_token_removal_preserves_mutation_and_user_output_positions():
    class Network(_Network):
        def __init__(self):
            super().__init__()
            self.register_buffer("total", torch.tensor(0.0))

        def forward(self, x):
            self.total.add_(x.sum())
            return super().forward(x)

    model = Network()
    x = torch.ones(4, 3)
    capture = capture_forward(model, (x,))
    assert capture.user_output_indices == (1,)
    (mutation,) = capture.mutations
    assert (mutation.input_index, mutation.output_index, mutation.target) == (
        1,
        0,
        "total",
    )
    outputs = capture.exported_program.graph_module(*capture.flat_inputs)
    torch.testing.assert_close(outputs[0], model.total + x.sum())
    torch.testing.assert_close(outputs[1], x @ model.weight @ model.weight)
