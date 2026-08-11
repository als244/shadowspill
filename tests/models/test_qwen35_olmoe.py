from __future__ import annotations

import torch

from models.pytorch import OLMoE, OLMoEConfig, Qwen35, Qwen35Config
from models.pytorch.qwen35 import _delta_rule, _delta_rule_reference


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


def test_qwen_delta_rule_bounded_operation_matches_reference_vjp() -> None:
    torch.manual_seed(29)
    shapes = ((1, 5, 2, 4), (1, 5, 4, 3), (1, 5, 4))
    reference_inputs = (
        torch.randn(shapes[0], requires_grad=True),
        torch.randn(shapes[0], requires_grad=True),
        torch.randn(shapes[1], requires_grad=True),
        torch.sigmoid(torch.randn(shapes[2])).requires_grad_(),
        (-torch.rand(shapes[2])).requires_grad_(),
    )
    bounded_inputs = tuple(
        value.detach().clone().requires_grad_() for value in reference_inputs
    )
    expected = _delta_rule_reference(*reference_inputs)
    actual = _delta_rule(*bounded_inputs)
    torch.testing.assert_close(actual, expected)
    tangent = torch.randn_like(expected)
    expected_gradients = torch.autograd.grad(expected, reference_inputs, tangent)
    actual_gradients = torch.autograd.grad(actual, bounded_inputs, tangent)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=2e-5, atol=2e-6
        )


def test_qwen_delta_rule_passes_pytorch_operation_contract_checks() -> None:
    torch.manual_seed(31)
    arguments = (
        torch.randn(1, 3, 2, 4, requires_grad=True),
        torch.randn(1, 3, 2, 4, requires_grad=True),
        torch.randn(1, 3, 4, 3, requires_grad=True),
        torch.sigmoid(torch.randn(1, 3, 4)).requires_grad_(),
        (-torch.rand(1, 3, 4)).requires_grad_(),
    )
    torch.library.opcheck(_delta_rule, arguments)


def test_tiny_olmoe_auxiliary_loss_reaches_router() -> None:
    config = OLMoEConfig(2, 32, 4, 4, 8, 4, 2, 16, 97, max_seq_len=16)
    model = OLMoE(config)
    tokens = torch.randint(0, 97, (1, 5))
    targets = torch.randint(0, 97, (1, 5))
    loss = model.loss(tokens, targets, aux_coef=0.01)
    loss.backward()
    assert loss.shape == ()
    assert all(block.moe.router.weight.grad is not None for block in model.blocks)
