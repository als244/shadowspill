"""EP masking uses the provider's routed logits, including packed router weights."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from shadowspill_workloads.models.catalog import entry


@pytest.mark.parametrize("family", ["qwen3moe", "qwen35moe", "olmoe"])
@pytest.mark.parametrize(
    "selected", [None, (True, False, True, False, True), (False,) * 5]
)
def test_padding_mask_preserves_router_precision_and_independent_gradients(
    family, selected, monkeypatch
):
    from mlops.expert_parallel.parameters import BF16ComputeWeight
    from mlops.expert_parallel.quack import router
    from mlops.kernels import router as router_math

    from shadowspill_workloads.models.olmoe.implementations.mlops import MoE as OLMoE
    from shadowspill_workloads.models.shared.qwen_moe.experts import MoE as QwenMoE

    config = entry(family + "-tiny").config()
    if family != "olmoe":
        config = replace(config, d_ff_shared=0)
    generator = torch.Generator().manual_seed(384)
    payload = torch.randn(
        config.n_experts, config.d_model, generator=generator
    ).bfloat16()

    class Experts(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                model_dim=config.d_model,
                top_k=config.top_k,
                renormalize_topk=family != "olmoe",
                router_weight_grad_dtype=torch.float32,
            )
            self.router_weight = nn.Parameter(
                payload.float() if family == "olmoe" else BF16ComputeWeight(payload)
            )

        def forward(self, hidden, *, expert_ids, routing_weights):
            self.ids = expert_ids.detach()
            return hidden * 0.25

    route_calls, projection_calls = [], []
    project = router_math.routed_logits

    def projection(*args, **kwargs):
        projection_calls.append(1)
        return project(*args, **kwargs)

    def route(logits, top_k, normalize):
        route_calls.append(1)
        chosen, ids = logits.topk(top_k, dim=-1)
        weights = chosen.softmax(-1) if normalize else logits.softmax(-1).gather(1, ids)
        counts = torch.bincount(ids.flatten(), minlength=logits.shape[1])
        return weights, ids, counts

    monkeypatch.setattr(router_math, "routed_logits", projection)
    monkeypatch.setattr(router, "route_op", route)
    experts = Experts()
    model = (OLMoE if family == "olmoe" else QwenMoE)(
        config, expert_factory=lambda: experts
    )
    hidden = (
        torch.randn(1, 5, config.d_model, generator=generator)
        .bfloat16()
        .requires_grad_()
    )
    residual = torch.zeros_like(hidden)
    mask = None if selected is None else torch.tensor([selected])
    options = {"return_metrics": True} if family == "olmoe" else {}
    result = model(hidden, residual, router_mask=mask, **options)
    output, counts, sums = result[0], result[-2], result[-1]
    expected_weight = payload.float().requires_grad_()
    expected_hidden = hidden.detach().clone().requires_grad_()
    logits = F.linear(expected_hidden.float().flatten(0, 1), expected_weight)
    ids = logits.topk(config.top_k, dim=-1).indices
    active = torch.ones(5, dtype=torch.bool) if mask is None else mask.flatten()
    expected_counts = F.one_hot(ids[active], config.n_experts).sum((0, 1))
    expected_sums = logits[active].softmax(-1).sum(0)
    torch.testing.assert_close(output, residual + hidden * 0.25, atol=0, rtol=0)
    torch.testing.assert_close(counts, expected_counts, atol=0, rtol=0)
    torch.testing.assert_close(sums, expected_sums, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(experts.ids, ids, atol=0, rtol=0)
    weights = torch.arange(config.n_experts, dtype=torch.float32)
    gradients = torch.autograd.grad(
        (sums * weights).sum(), (hidden, experts.router_weight)
    )
    expected = torch.autograd.grad(
        (expected_sums * weights).sum(), (expected_hidden, expected_weight)
    )
    torch.testing.assert_close(gradients, expected, atol=2e-6, rtol=3e-5)
    assert len(projection_calls) == len(route_calls) == 1
