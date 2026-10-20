"""Compressed KV projection absorption preserves the LoRA branch and gradients."""

import copy

import pytest
import torch
from torch import nn

pytest.importorskip("mlops")
from mlops.lora import LoRAConfig
from mlops.sequential_moe.precision.linear import FrozenLinear, FrozenLoRALinear
from mlops.sequential_moe.precision.storage import pack

from tests.precision import require_fp8_gemms
from workloads.models.glm53_flash.config import Config
from workloads.models.glm53_flash.implementations.mlops.attention import SparseMLA


class CausalKeys(nn.Module):
    def forward(self, x, q_residual, boundaries):
        rows = torch.arange(x.shape[0], device=x.device)
        keys = rows.expand(x.shape[0], -1)
        return torch.where(keys <= rows[:, None], keys, -1).to(torch.int32)


def expanded(model, x):
    c = model.config
    query = model.q_b_proj(model.q_a_layernorm(model.q_a_proj(x))).reshape(
        -1, c.num_attention_heads, c.qk_nope_head_dim
    )
    latent = model.kv_a_layernorm(model.kv_a_proj_with_mqa(x))
    key, value = (
        model.kv_b_proj(latent)
        .reshape(-1, c.num_attention_heads, c.qk_nope_head_dim + c.v_head_dim)
        .split((c.qk_nope_head_dim, c.v_head_dim), -1)
    )
    scores = torch.einsum("thd,shd->hts", query.float(), key.float())
    scores = scores * c.qk_nope_head_dim**-0.5
    future = torch.ones(x.shape[0], x.shape[0], device=x.device).triu(1).bool()
    probability = scores.masked_fill(future, -torch.inf).softmax(-1)
    output = torch.einsum("hts,shd->thd", probability, value.float()).to(x.dtype)
    return model.o_proj(output.flatten(-2))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_fp8_kv_projection_matches_expanded_attention_and_gradients(lora, compiled):
    require_fp8_gemms()
    torch.manual_seed(418)
    model = SparseMLA(Config.tiny(hidden_size=128), "cuda")
    parts, spec = pack(model.kv_b_proj.weight.detach(), "fp8")
    projection = FrozenLinear(parts, spec)
    if lora:
        projection = FrozenLoRALinear(
            projection, LoRAConfig(rank=8, alpha=8, factor_dtype=torch.float32)
        )
        # Nonzero B is necessary to test both factor gradients and both
        # absorbed key/value contributions, rather than only the base model.
        with torch.no_grad():
            projection.lora_b.normal_(std=0.02)
    model.kv_b_proj = projection
    model.indexer = CausalKeys()
    reference = copy.deepcopy(model)
    x = torch.randn(13, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    fn = torch.compile(model, fullgraph=True) if compiled else model
    actual = fn(x, (0, 13), None, None)
    expected = expanded(reference, xr)
    dy = torch.randn_like(actual)
    inputs = (x, projection.lora_a, projection.lora_b) if lora else (x,)
    targets = (
        (xr, reference.kv_b_proj.lora_a, reference.kv_b_proj.lora_b) if lora else (xr,)
    )
    gradients = torch.autograd.grad(actual, inputs, dy)
    reference_gradients = torch.autograd.grad(expected, targets, dy)
    for got, want in zip(
        (actual, *gradients), (expected, *reference_gradients), strict=True
    ):
        error = (got.float() - want.float()).norm() / want.float().norm().clamp_min(
            1e-20
        )
        assert float(error.detach()) < 0.015, float(error.detach())
    assert all(not p.requires_grad for p in projection.components)
    assert not hasattr(projection, "weight")
