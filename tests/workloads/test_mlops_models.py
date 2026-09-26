from __future__ import annotations

import pytest
import torch

from workloads.mlops import Llama3 as MlopsLlama3
from workloads.mlops import OLMoE as MlopsOLMoE
from workloads.mlops import Qwen35 as MlopsQwen35
from workloads.pytorch import (
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


def _loss_and_gradients(
    model: torch.nn.Module, tokens: torch.Tensor, targets: torch.Tensor, lengths: object
) -> tuple[torch.Tensor, ...]:
    model.zero_grad()
    loss = model.loss(tokens, targets, seq_lens=lengths)
    loss.backward()
    return (loss.detach(), *(parameter.grad for parameter in model.parameters()))


@pytest.mark.parametrize(
    "build",
    [
        lambda: MlopsLlama3(Llama3Config(2, 32, 4, 2, 64, 97, max_seq_len=16)),
        lambda: MlopsQwen35(
            Qwen35Config(4, 32, 4, 4, 2, 8, 0.5, 2, 4, 4, 4, 3, 64, 97, max_seq_len=16)
        ),
        lambda: MlopsOLMoE(OLMoEConfig(2, 32, 4, 4, 8, 4, 2, 16, 97, max_seq_len=16)),
    ],
    ids=["llama3", "qwen35", "olmoe"],
)
def test_a_lengths_tensor_packs_as_the_same_integers(build: object) -> None:
    # The same three sequences, once as structure and once as data padded
    # with empty sequences to a fixed size, as a packed microbatch carries them.
    torch.manual_seed(29)
    model = build()  # type: ignore[operator]
    tokens = torch.randint(0, 97, (1, 12))
    targets = torch.randint(0, 97, (1, 12))
    as_integers = _loss_and_gradients(model, tokens, targets, (5, 3, 4))
    as_tensor = _loss_and_gradients(
        model, tokens, targets, torch.tensor([5, 3, 4, 0, 0], dtype=torch.int32)
    )
    for got, want in zip(as_tensor, as_integers, strict=True):
        torch.testing.assert_close(got, want)
