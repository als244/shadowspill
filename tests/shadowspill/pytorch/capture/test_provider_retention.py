"""Opaque providers declare saved forward state without engine name checks."""

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from shadowspill.errors import CaptureError
from shadowspill.pytorch import RetentionPolicy, RetentionRule
from shadowspill.pytorch.capture.aot import capture_graph_pair


@torch.library.custom_op("shadowspill_test_retention::selected_value", mutates_args=())
def selected_value(value: torch.Tensor) -> torch.Tensor:
    return value.exp()


@selected_value.register_fake
def selected_value_fake(value):
    return torch.empty_like(value)


def setup_context(ctx, inputs, output):
    ctx.save_for_backward(output)


def backward(ctx, gradient):
    (selected,) = ctx.saved_tensors
    return gradient * selected


selected_value.register_autograd(backward, setup_context=setup_context)
TARGET = torch.ops.shadowspill_test_retention.selected_value.default


def graph_and_values():
    values = (torch.linspace(-0.5, 0.5, 7, dtype=torch.float64, requires_grad=True),)
    return make_fx(lambda x: selected_value(x).sum())(*values), values


@pytest.mark.parametrize("declared", [False, True])
@pytest.mark.parametrize("override", [None, "retain", "recompute"])
def test_provider_default_and_explicit_override_preserve_vjp(
    monkeypatch, declared, override
):
    monkeypatch.setattr(
        TARGET, "shadowspill_retain_for_backward", declared, raising=False
    )
    graph, values = graph_and_values()
    policy = RetentionPolicy(
        default="recompute",
        rules=() if override is None else (RetentionRule(str(TARGET), override),),
    )
    pair = capture_graph_pair(
        graph, values, original_output=graph(*values), memory_budget=0, retention=policy
    )
    retain = declared if override is None else override == "retain"
    assert (str(TARGET) in pair.backward.operator_targets) is not retain
    with torch.no_grad():
        prepared = pair.forward.graph_module(*values)
        (gradient,) = pair.backward.graph_module(
            *prepared[-pair.saved_value_count :], torch.ones((), dtype=torch.float64)
        )
    torch.testing.assert_close(gradient, values[0].exp(), rtol=1e-12, atol=1e-12)


def test_provider_declaration_changes_capture_identity(monkeypatch):
    graph, _ = graph_and_values()
    policy = RetentionPolicy(default="recompute")
    before = policy.digest(graph)
    assert "forward_choices" not in policy.capture_identity(graph)
    monkeypatch.setattr(TARGET, "shadowspill_retain_for_backward", True, raising=False)
    assert policy.digest(graph) != before
    assert policy.capture_identity(graph)["forward_choices"] == [str(TARGET)]


@pytest.mark.parametrize("invalid", ["retain", 1, None])
def test_provider_declaration_requires_static_bool(monkeypatch, invalid):
    monkeypatch.setattr(
        TARGET, "shadowspill_retain_for_backward", invalid, raising=False
    )
    graph, _ = graph_and_values()
    with pytest.raises(CaptureError, match="must be a static bool"):
        RetentionPolicy().capture_identity(graph)
