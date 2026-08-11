from __future__ import annotations

import torch

from models.mlops import Llama3 as MlopsLlama3
from models.mlops import OLMoE as MlopsOLMoE
from models.mlops import Qwen35 as MlopsQwen35
from models.pytorch import (
    Llama3,
    Llama3Config,
    OLMoE,
    OLMoEConfig,
    Qwen35,
    Qwen35Config,
)


def _assert_state_dict_compatible(
    reference: torch.nn.Module, optimized: torch.nn.Module
) -> None:
    assert reference.state_dict().keys() == optimized.state_dict().keys()
    optimized.load_state_dict(reference.state_dict(), strict=True)


def _assert_loss_and_gradients(
    reference: torch.nn.Module,
    optimized: torch.nn.Module,
    tokens: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    reference_loss = reference.loss(tokens, targets)
    optimized_loss = optimized.loss(tokens, targets)
    torch.testing.assert_close(optimized_loss, reference_loss)
    reference_loss.backward()
    optimized_loss.backward()
    for reference_parameter, optimized_parameter in zip(
        reference.parameters(), optimized.parameters(), strict=True
    ):
        torch.testing.assert_close(optimized_parameter.grad, reference_parameter.grad)


def test_tiny_llama_matches_external_operations() -> None:
    torch.manual_seed(17)
    config = Llama3Config(2, 32, 4, 2, 64, 97, max_seq_len=16)
    reference = Llama3(config)
    optimized = MlopsLlama3(config)
    _assert_state_dict_compatible(reference, optimized)
    tokens = torch.randint(0, config.vocab_size, (1, 6))
    torch.testing.assert_close(optimized(tokens), reference(tokens))
    _assert_loss_and_gradients(
        reference,
        optimized,
        tokens,
        torch.randint(0, config.vocab_size, tokens.shape),
    )


def test_tiny_qwen_matches_external_operations() -> None:
    torch.manual_seed(19)
    config = Qwen35Config(4, 32, 4, 4, 2, 8, 0.5, 2, 4, 4, 4, 3, 64, 97, max_seq_len=16)
    reference = Qwen35(config)
    optimized = MlopsQwen35(config)
    _assert_state_dict_compatible(reference, optimized)
    tokens = torch.randint(0, config.vocab_size, (1, 6))
    torch.testing.assert_close(optimized(tokens), reference(tokens))
    _assert_loss_and_gradients(
        reference,
        optimized,
        tokens,
        torch.randint(0, config.vocab_size, tokens.shape),
    )


def test_tiny_olmoe_matches_external_operations() -> None:
    torch.manual_seed(23)
    config = OLMoEConfig(2, 32, 4, 4, 8, 4, 2, 16, 97, max_seq_len=16)
    reference = OLMoE(config)
    optimized = MlopsOLMoE(config)
    _assert_state_dict_compatible(reference, optimized)
    tokens = torch.randint(0, config.vocab_size, (1, 6))
    torch.testing.assert_close(optimized(tokens), reference(tokens))
    _assert_loss_and_gradients(
        reference,
        optimized,
        tokens,
        torch.randint(0, config.vocab_size, tokens.shape),
    )
