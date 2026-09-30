"""Returned routing/loss summaries must leave training math unchanged."""

import pytest
import torch

from tests.precision import low_precision_dtype
from workloads.mlops import OLMoE, OLMoEConfig


@pytest.mark.cuda
def test_olmoe_returned_metrics_preserve_loss_and_gradients():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for mlops MoE")
    torch.manual_seed(141)
    config = OLMoEConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        n_experts=4,
        top_k=2,
        d_ff_expert=128,
        vocab_size=256,
        max_seq_len=128,
    )
    model = OLMoE(config).to(device="cuda", dtype=low_precision_dtype())
    tokens = torch.randint(0, 256, (1, 128), device="cuda")
    targets = torch.randint(0, 256, (1, 128), device="cuda")
    targets[:, -8:] = -100
    plain = model.loss(tokens, targets, aux_coef=0.01, reduction="sum")
    observed, metrics = model.loss(
        tokens, targets, aux_coef=0.01, reduction="sum", return_metrics=True
    )
    torch.testing.assert_close(plain, observed, rtol=1e-6, atol=1e-5)
    plain_grads = torch.autograd.grad(plain, tuple(model.parameters()))
    observed_grads = torch.autograd.grad(observed, tuple(model.parameters()))
    for expected, actual in zip(plain_grads, observed_grads, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
    assert all(
        isinstance(value, torch.Tensor) and not value.requires_grad
        for value in metrics.values()
    )
    torch.testing.assert_close(
        metrics["ce_sum"] + metrics["weighted_auxiliary_sum"], observed.detach()
    )
    assert metrics["trained_tokens"].item() == 120
    # Counts include all 128 router rows, even the eight ignored loss targets.
    assert metrics["expert_counts"].shape == (2, 4)
    assert metrics["expert_counts"].sum(dim=1).tolist() == [256, 256]
