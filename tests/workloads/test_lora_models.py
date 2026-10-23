"""Whole-model LoRA losses, gradients, updates, loading and frozen-state checks."""

import contextlib
import copy
from dataclasses import replace
from importlib import import_module
from unittest.mock import patch

import pytest
import torch
from mlops.dispatch import use_implementations

from workloads.lora import configure_lora
from workloads.models.shared.routed_lora import expert_lora as reference_expert_lora

#: Where the mlops MoE modules take their routed-expert LoRA kernel from.
ROUTED_KERNELS = (
    "workloads.models.olmoe.implementations.mlops.expert_lora",
    "workloads.models.shared.qwen_moe.experts.expert_lora",
)


@contextlib.contextmanager
def reference_experts():
    """The routed experts' LoRA computed by the ordinary-PyTorch reference."""
    with contextlib.ExitStack() as stack:
        for target in ROUTED_KERNELS:
            stack.enter_context(patch(target, reference_expert_lora))
        yield


MODELS = [
    (impl, family)
    for impl in ("pytorch", "mlops")
    for family in ("llama3", "qwen3", "qwen35", "olmoe")
]
MODELS += [("mlops", family) for family in ("qwen3moe", "qwen35moe")]


def family_classes(implementation, family):
    """The family's ``Config`` and the implementation's ``Model``."""
    config = import_module(f"workloads.models.{family}.config").Config
    model = import_module(
        f"workloads.models.{family}.implementations.{implementation}"
    ).Model
    return config, model


def tiny_model(implementation, family):
    config_type, model_type = family_classes(implementation, family)
    if family == "llama3":
        config = config_type(2, 32, 4, 2, 64, 97, max_seq_len=16)
    elif family == "qwen3":
        config = config_type(2, 32, 4, 2, 16, 64, 97, max_seq_len=16)
    elif family == "qwen35":
        config = config_type(
            4, 32, 4, 4, 2, 8, 0.5, 2, 4, 4, 4, 3, 64, 97, max_seq_len=16
        )
    elif family == "olmoe":
        config = config_type(2, 32, 4, 2, 8, 4, 2, 16, 97, max_seq_len=16)
    else:
        config = config_type()
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
    return model_type(config)


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
    # FP32 factors: the reference comparison below holds FP32 tolerances.
    configure_lora(model, rank=4, alpha=4, head=head, factor_dtype="float32")
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
            reference_experts(),
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
        tiny_model(implementation, family),
        rank=4,
        alpha=4,
        head=head,
        factor_dtype="float32",
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
    from workloads.models.construction import initialize

    config_type, model_type = family_classes("mlops", "qwen35")
    config = replace(
        config_type(4, 32, 4, 4, 2, 8, 0.5, 2, 4, 4, 4, 3, 64, 97, max_seq_len=16),
        tied_embeddings=True,
    )
    with torch.device("meta"):
        model = configure_lora(
            model_type(config), rank=4, head="lora", trainable_base=["embed.weight"]
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
        tiny_model("mlops", "qwen35moe"), rank=4, shared_experts=True
    )
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    assert any(".moe.shared." in name and "lora_" in name for name in names)
    assert all("lora_" in name for name in names)
    assert not any(".shared.gate.lora_" in name for name in names)


@pytest.mark.parametrize("implementation,family", MODELS)
def test_text_recipe_materializes_mixed_precision_factors(implementation, family):
    from shadowspill.training._model import initialize_model
    from workloads.models.construction import build_on_meta, initialize

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


@pytest.mark.parametrize(
    "implementation,family",
    [("mlops", "qwen3moe"), ("mlops", "olmoe"), ("pytorch", "olmoe")],
)
def test_routed_factors_follow_the_declared_shapes(implementation, family):
    # A caller that partitions parameters as their modules are constructed
    # leaves the expert banks without their data; the factors are sized from
    # what the module declares all the same.
    model = tiny_model(implementation, family)
    routed = [module for module in model.modules() if hasattr(module, "expert_shapes")]
    assert routed
    for module in routed:
        for parameter in module.parameters(recurse=False):
            parameter.data = torch.empty(0)
    configure_lora(model, rank=4, alpha=4)
    for module in routed:
        (experts, gate_up_out, gate_up_in), (_, down_out, down_in) = (
            module.expert_shapes
        )
        assert module.lora_gate_up_a.shape == (experts, 4, gate_up_in)
        assert module.lora_gate_up_b.shape == (experts, gate_up_out, 4)
        assert module.lora_down_a.shape == (experts, 4, down_in)
        assert module.lora_down_b.shape == (experts, down_out, 4)
        banks = [
            parameter
            for name, parameter in module.named_parameters(recurse=False)
            if not name.startswith("lora_")
        ]
        assert banks and not any(bank.requires_grad for bank in banks)
        assert module.lora_gate_up_a.requires_grad


@pytest.mark.parametrize("implementation,family", MODELS)
def test_family_presets_exclude_control_projections(implementation, family):
    model = tiny_model(implementation, family)
    # A new small control projection must not be silently picked up by defaults.
    model.control_probe = torch.nn.Linear(32, 2, bias=False)
    configure_lora(model, rank=4, alpha=4)
    assert model.LORA_TARGETS
    assert not hasattr(model.control_probe, "lora_config")
    assert not model.control_probe.weight.requires_grad
    for name, module in model.named_modules():
        if name.endswith("w_ba"):
            assert not hasattr(module, "lora_config")
            assert not module.weight.requires_grad
        if name.endswith("w_qkvz"):
            assert hasattr(module, "lora_config")
            assert module.lora_b.shape[0] == module.out_features


def test_qwen35_explicit_targets_can_override_family_defaults():
    model = configure_lora(
        tiny_model("mlops", "qwen35"), rank=4, targets=["blocks.0.mixer.w_ba"]
    )
    assert hasattr(model.blocks[0].mixer.w_ba, "lora_config")
    assert not hasattr(model.blocks[0].mixer.w_qkvz, "lora_config")
