"""Whole-model LoRA losses, gradients, updates, loading and frozen-state checks."""

import copy
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from mlops.dispatch import use_implementations

from workloads import mlops, pytorch
from workloads.lora import configure_lora
from workloads.lora.reference import expert_lora as reference_expert_lora

MODELS = [
    (impl, family)
    for impl in ("pytorch", "mlops")
    for family in ("Llama3", "Qwen35", "OLMoE")
]
MODELS += [("mlops", family) for family in ("Qwen3MoE", "Qwen35MoE")]


def tiny_model(implementation, family):
    models = mlops if implementation == "mlops" else pytorch
    if family == "Llama3":
        config = models.Llama3Config(2, 32, 4, 2, 64, 97, max_seq_len=16)
    elif family == "Qwen35":
        config = models.Qwen35Config(
            4, 32, 4, 4, 2, 8, 0.5, 2, 4, 4, 4, 3, 64, 97, max_seq_len=16
        )
    elif family == "OLMoE":
        config = models.OLMoEConfig(2, 32, 4, 2, 8, 4, 2, 16, 97, max_seq_len=16)
    else:
        config = getattr(models, family + "Config")()
        config = replace(
            config,
            n_layers=4,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            head_dim=8,
            n_experts=4,
            top_k=2,
            d_ff_expert=16,
            d_ff_shared=16 if config.d_ff_shared else 0,
            vocab_size=97,
            max_seq_len=16,
            lin_k_heads=2,
            lin_v_heads=4,
            lin_k_head_dim=4,
            lin_v_head_dim=4,
        )
    return getattr(models, family)(config)


@pytest.mark.parametrize("implementation,family", MODELS)
@pytest.mark.parametrize("head", ["frozen", "lora"])
def test_full_model_losses_gradients_updates_and_frozen_state(
    implementation, family, head
):
    torch.manual_seed(641)
    model = tiny_model(implementation, family)
    tokens = torch.randint(0, 97, (1, 7))
    targets = torch.randint(0, 97, (1, 7))
    before_logits = model(tokens, (3, 4)).detach()
    original = {name: p.detach().clone() for name, p in model.named_parameters()}
    configure_lora(model, rank=4, alpha=4, head=head)
    torch.testing.assert_close(
        model(tokens, (3, 4)), before_logits, atol=2e-6, rtol=3e-5
    )
    # Nonzero B exercises both factor gradients, including the routed down path.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_" in name and name.endswith("b"):
                parameter.normal_(std=0.03)
    oracle = copy.deepcopy(model)
    selected = [p for p in model.parameters() if p.requires_grad]
    reference_selected = [p for p in oracle.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(selected, lr=0.03)
    reference_optimizer = torch.optim.SGD(reference_selected, lr=0.03)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        actual = model.loss(tokens, targets, seq_lens=(3, 4))
        with (
            patch("workloads.lora.experts.expert_lora", reference_expert_lora),
            use_implementations(
                {
                    "head_loss": "native_torch.head_loss",
                    "lora_head_loss": "native_torch.lora_head_loss",
                }
            ),
        ):
            expected = oracle.loss(tokens, targets, seq_lens=(3, 4))
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=3e-5)
        actual.backward()
        expected.backward()
        for actual_p, expected_p in zip(selected, reference_selected, strict=True):
            assert actual_p.grad is not None and torch.isfinite(actual_p.grad).all()
            torch.testing.assert_close(
                actual_p.grad, expected_p.grad, atol=3e-5, rtol=3e-4
            )
        optimizer.step()
        reference_optimizer.step()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            assert parameter.grad is None
            torch.testing.assert_close(parameter, original[name], atol=0, rtol=0)
    torch.testing.assert_close(
        model.state_dict(), oracle.state_dict(), atol=3e-6, rtol=3e-5
    )
    restored = configure_lora(
        tiny_model(implementation, family), rank=4, alpha=4, head=head
    )
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(restored(tokens), model(tokens))


@pytest.mark.parametrize("implementation,family", MODELS)
def test_whole_model_aot_capture(implementation, family):
    torch._dynamo.reset()
    torch.manual_seed(418)
    model = configure_lora(
        tiny_model(implementation, family), rank=4, alpha=4, head="lora"
    )
    oracle = copy.deepcopy(model)
    tokens, targets = torch.randint(0, 97, (1, 7)), torch.randint(0, 97, (1, 7))
    actual = torch.compile(model.loss, backend="aot_eager", fullgraph=True)(
        tokens, targets
    )
    expected = oracle.loss(tokens, targets)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(actual, expected)
    for p, q in zip(model.parameters(), oracle.parameters(), strict=True):
        if p.requires_grad:
            torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=2e-4)
        else:
            assert p.grad is None


def test_meta_initialization_tied_head_and_full_parameter_override():
    from workloads.recipes.text.models import initialize

    config = replace(
        pytorch.Qwen35Config(
            4, 32, 4, 4, 2, 8, 0.5, 2, 4, 4, 4, 3, 64, 97, max_seq_len=16
        ),
        tied_embeddings=True,
    )
    with torch.device("meta"):
        model = configure_lora(
            mlops.Qwen35(config), rank=4, head="lora", trainable_base=["embed.weight"]
        )
    assert model.embed.weight is model.lm_head.weight
    from shadowspill.training._model import initialize_model

    initialize_model(model, initialize=initialize)
    assert model.lm_head.weight is model.embed.weight
    assert model.embed.weight.requires_grad
    assert torch.isfinite(model.lm_head.lora_a).all()
    assert not model.lm_head.lora_b.count_nonzero()
    tokens = torch.randint(0, 97, (1, 6))
    model.loss(tokens, tokens).backward()
    assert model.embed.weight.grad is not None


def test_shared_expert_lora_is_explicit():
    model = configure_lora(
        tiny_model("mlops", "Qwen35MoE"), rank=4, shared_experts=True
    )
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    assert any(".moe.shared." in name and "lora_" in name for name in names)
    assert all("lora_" in name for name in names)


@pytest.mark.parametrize("implementation,family", MODELS)
def test_text_recipe_materializes_mixed_precision_factors(implementation, family):
    from shadowspill.training._model import initialize_model
    from workloads.recipes.text.models import build_on_meta, initialize

    original_dtype = torch.get_default_dtype()
    model = build_on_meta(
        lambda: tiny_model(implementation, family),
        dtype="bfloat16",
        lora={"rank": 4, "alpha": 4, "head": "lora", "factor_dtype": "float32"},
    )
    assert torch.get_default_dtype() == original_dtype
    assert all(p.device.type == "meta" for p in model.parameters())
    initialize_model(model, initialize=initialize)
    for name, parameter in model.named_parameters():
        if "lora_" in name:
            assert parameter.dtype == torch.float32 and parameter.requires_grad
            assert torch.isfinite(parameter).all()
        else:
            assert not parameter.requires_grad
    assert model.embed.weight.dtype == torch.bfloat16
    tokens = torch.randint(0, 97, (1, 7))
    loss = model.loss(tokens, tokens, seq_lens=(3, 4))
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
