from __future__ import annotations

import torch

from models.pytorch import OLMoE, OLMoEConfig, Qwen35, Qwen35Config


def test_reduced_family_parameter_counts() -> None:
    with torch.device("meta"):
        qwen = Qwen35(Qwen35Config.numerical())
        olmoe = OLMoE(OLMoEConfig.numerical())
    assert sum(parameter.numel() for parameter in qwen.parameters()) == 1_006_955_408
    assert sum(parameter.numel() for parameter in olmoe.parameters()) == 975_766_528


def test_tiny_qwen_hybrid_schedule_and_gradients() -> None:
    config = Qwen35Config(
        4,
        32,
        4,
        4,
        2,
        8,
        0.5,
        2,
        4,
        4,
        4,
        3,
        64,
        97,
        max_seq_len=16,
    )
    model = Qwen35(config)
    tokens = torch.randint(0, 97, (1, 5))
    targets = torch.randint(0, 97, (1, 5))
    loss = model.loss(tokens, targets)
    loss.backward()
    assert tuple(block.kind for block in model.blocks) == (
        "linear",
        "linear",
        "linear",
        "full",
    )
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_tiny_olmoe_auxiliary_loss_reaches_router() -> None:
    config = OLMoEConfig(2, 32, 4, 4, 8, 4, 2, 16, 97, max_seq_len=16)
    model = OLMoE(config)
    tokens = torch.randint(0, 97, (1, 5))
    targets = torch.randint(0, 97, (1, 5))
    loss = model.loss(tokens, targets, aux_coef=0.01)
    loss.backward()
    assert loss.shape == ()
    assert all(block.moe.router.weight.grad is not None for block in model.blocks)
