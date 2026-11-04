"""Discrete forward choices survive both checkpoint variants by default."""

import json

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from shadowspill.pytorch import GraphPairPolicy, RetentionPolicy, RetentionRule
from shadowspill.pytorch.capture.aot import capture_graph_pair
from shadowspill.pytorch.graph_pairs.serialization import CachedAotGraphPair

ATTENTION = (RetentionRule("mlops.flash_attention_builtin_aten_fwd.default", "retain"),)
POLICIES = (
    GraphPairPolicy(),
    GraphPairPolicy(recompute=RetentionPolicy(default="recompute", rules=ATTENTION)),
    GraphPairPolicy(save=RetentionPolicy(default="recompute", rules=ATTENTION)),
)


def inputs():
    generator = torch.Generator().manual_seed(731)
    return tuple(
        torch.randn(shape, generator=generator, dtype=torch.float64).requires_grad_()
        for shape in ((13, 7), (7, 5), (5, 3))
    )


def forward(provider, x, w, experts):
    logits = x @ w
    if provider == "mlops":
        from mlops.lora.routing import routing

        weights, ids = routing(logits, 2, "topk_then_softmax")
    else:
        values, ids = logits.topk(2, dim=-1)
        weights = values.softmax(-1)
    return (experts[ids.long()] * weights[:, :, None]).sum(1)


def reference(provider, x, w, experts):
    logits = x @ w
    # MLOps computes selected scores/weights in FP32, including its CPU path.
    values, ids = (logits.float() if provider == "mlops" else logits).topk(2, dim=-1)
    return (experts[ids] * values.softmax(-1)[:, :, None]).sum(1)


def capture(provider, policy, variant):
    values = inputs()

    def function(*args):
        return forward(provider, *args)

    graph = make_fx(function)(*values)
    pair = capture_graph_pair(
        graph,
        values,
        original_output=function(*values),
        memory_budget=0.0 if variant == "recompute" else 1.0,
        retention=policy.variant(variant),
    )
    return values, graph, pair


@pytest.mark.parametrize("provider", ["aten", "mlops"])
@pytest.mark.parametrize("variant", ["save", "recompute"])
@pytest.mark.parametrize("policy", POLICIES)
def test_all_search_policies_keep_choices_and_correct_vjps(provider, variant, policy):
    values, graph, pair = capture(provider, policy, variant)
    target = "aten.topk.default" if provider == "aten" else "mlops.lora_route.default"
    assert target not in pair.backward.operator_targets
    assert target not in pair.retention.regenerated_operators
    identity = policy.capture_identity(graph)
    assert identity["forward_choices"] == {"save": [target], "recompute": [target]}
    tangent = torch.arange(39, dtype=torch.float64).reshape(13, 3) / 39
    expected = reference(provider, *values)
    gradients = torch.autograd.grad(expected, values, tangent)
    with torch.no_grad():
        prepared = pair.forward.graph_module(*values)
        saved = prepared[-pair.saved_value_count :]
        actual = pair.backward.graph_module(*saved, tangent)
    dtype = torch.int64 if provider == "aten" else torch.int32
    assert any(
        tuple(v.shape) == (13, 2) and v.dtype == dtype
        for v in saved
        if isinstance(v, torch.Tensor)
    )
    if provider == "mlops":
        assert any(
            tuple(v.shape) == (13, 2) and v.dtype == torch.float32
            for v in saved
            if isinstance(v, torch.Tensor)
        )
    tolerance = 3e-6 if provider == "mlops" else 1e-11
    torch.testing.assert_close(prepared[0], expected, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(tuple(actual), gradients, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("provider", ["aten", "mlops"])
def test_explicit_replay_overrides_default_and_changes_capture_identity(provider):
    target = "aten.topk.default" if provider == "aten" else "mlops.lora_route.default"
    policy = GraphPairPolicy(
        recompute=RetentionPolicy(
            default="recompute", rules=(RetentionRule(target, "recompute"),)
        )
    )
    _, graph, pair = capture(provider, policy, "recompute")
    assert target in pair.backward.operator_targets
    assert policy.digest(graph) != GraphPairPolicy().digest(graph)
    # The new built-in default also distinguishes this source from legacy cuts
    # that only keyed the cost class (which omitted ATen entirely).
    current = GraphPairPolicy().capture_identity(graph)
    legacy = {key: value for key, value in current.items() if key != "forward_choices"}
    assert json.dumps(current, sort_keys=True) != json.dumps(legacy, sort_keys=True)


def test_router_is_called_once_and_saved_choices_survive_archive(monkeypatch):
    from mlops.lora.routing import _route

    values, _, pair = capture("mlops", GraphPairPolicy(), "recompute")
    # A graph-pair archive restores the captured cuts; no policy callback runs.
    restored = CachedAotGraphPair.capture(pair).restore()
    assert restored.backward.compatibility_digest == pair.backward.compatibility_digest
    original = _route._backend_fns[None]
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setitem(_route._backend_fns, None, counted)
    with torch.no_grad():
        prepared = restored.forward.graph_module(*values)
        restored.backward.graph_module(
            *prepared[-restored.saved_value_count :],
            torch.ones(13, 3, dtype=torch.float64),
        )
    assert len(calls) == 1


@pytest.mark.parametrize("provider", ["aten", "mlops"])
def test_small_replay_perturbation_cannot_change_expert_gradient_branch(provider):
    values = inputs()
    x, weight, _ = values
    with torch.no_grad():
        x.fill_(1)
        weight.zero_()
        weight[0].copy_(torch.tensor([3, 2, 2 - 1e-6, 0, -1]))
    tangent = torch.ones(13, 3, dtype=torch.float64)
    wanted = torch.autograd.grad(reference(provider, *values), values, tangent)[-1]
    target = "aten.topk.default" if provider == "aten" else "mlops.lora_route.default"

    def function(*args):
        return forward(provider, *args)

    observed = []
    for replay in (False, True):
        policy = RetentionPolicy(
            default="recompute",
            rules=(RetentionRule(target, "recompute"),) if replay else (),
        )
        graph = make_fx(function)(*values)
        pair = capture_graph_pair(
            graph,
            values,
            original_output=function(*values),
            memory_budget=0,
            retention=policy,
        )
        with torch.no_grad():
            prepared = pair.forward.graph_module(*values)
            saved = list(prepared[-pair.saved_value_count :])
            replacements = 0
            for index, value in enumerate(saved):
                if isinstance(value, torch.Tensor) and value.shape == weight.shape:
                    # Simulate slightly different replayed logits without
                    # changing the saved forward choices. Only the expert-bank
                    # gradient is compared: other gradients depend on weight.
                    value = value.clone()
                    value[0, 2].add_(2e-6)
                    saved[index] = value
                    replacements += 1
            assert replacements == 1
            observed.append(pair.backward.graph_module(*saved, tangent)[-1])
    torch.testing.assert_close(observed[0], wanted, rtol=3e-6, atol=3e-6)
    assert not torch.allclose(observed[1], wanted, rtol=1e-3, atol=1e-3)
