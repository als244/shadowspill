"""Dense Qwen reuses packed setup and token objectives without expert resources."""

from dataclasses import asdict, replace
from importlib import import_module

import pytest
import torch

from shadowspill.training.definitions import load_registry
from shadowspill.training.preparation.capture.templates import capture_inventory
from shadowspill.training.preparation.configuration.layouts import prepare_layouts
from shadowspill.training.preparation.setup import resolve_training
from shadowspill_client import (
    BatchConfig,
    BuildCandidates,
    LoRA,
    Memory,
    Model,
    Objective,
    PlanningOptions,
    Preparation,
    Weights,
)
from shadowspill_workloads.lora import configure_lora
from shadowspill_workloads.models.qwen3.training import Settings as Qwen3Settings
from shadowspill_workloads.models.qwen35.training import Settings as Qwen35Settings
from workload_tests.definitions.test_llama_definition import bound, datums

FAMILIES = ("qwen3", "qwen35")


def settings(family, *, dtype="float32", lora=False, tied=False):
    catalog = import_module(f"shadowspill_workloads.models.{family}.config")
    overrides = dict(
        n_layers=2,
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        head_dim=8,
        d_ff=48,
        vocab_size=97,
        max_seq_len=8,
        tied_embeddings=tied,
    )
    if family == "qwen35":
        overrides.update(
            full_attention_interval=2,
            lin_k_heads=2,
            lin_v_heads=4,
            lin_k_head_dim=4,
            lin_v_head_dim=4,
            partial_rotary_factor=0.5,
            zero_centered_norm=True,
        )
    config = replace(catalog.PRESETS["tiny"], **overrides)
    return Model(
        family,
        config={"architecture": asdict(config)},
        precision={"compute_dtype": dtype, "lora_dtype": "float32"},
        lora=LoRA(2) if lora else None,
        initialize=Weights.random(seed=17),
    )


def prepared(family, *, lora=False, dtype="float32", tied=False):
    registry = load_registry(["shadowspill-workloads"])
    setup = resolve_training(
        registry,
        model=settings(family, lora=lora, dtype=dtype, tied=tied),
        objective=Objective("cross_entropy"),
        batch=BatchConfig(
            limits={"token_capacity": 8, "max_sequence_length": 4, "max_sequences": 4}
        ),
        memory=Memory(1, 1),
        operations={
            "forward_backward": ["metrics", "logprobs"],
            "forward": ["logprobs"],
        },
        prepare=Preparation(
            candidates=BuildCandidates(microbatch_sizes=[8]),
            planning=PlanningOptions(microbatch_counts=[1]),
        ),
    )
    return registry, setup, prepare_layouts(setup)


@pytest.mark.parametrize("family", FAMILIES)
def test_all_presets_resolve_and_dense_build_has_no_expert_provider(family):
    registry, setup, layouts = prepared(family)
    catalog = import_module(f"shadowspill_workloads.models.{family}.config")
    for preset, config in catalog.PRESETS.items():
        resolved = registry.resolve_model(family, config={"preset": preset})
        assert resolved.config["architecture"] == asdict(config)
        assert resolved.definition.capabilities == ("target_logprobs",)
    assert resolve_training(registry, setup.config).config == setup.config
    for template in capture_inventory(layouts).templates.values():
        build = template.model_build
        assert not build.operator_preparation and not build.resources
        assert not any(
            "moe" in name or "lora_route" in name
            for name in build.allocation_invariants
        )
        assert any("gated_delta" in name for name in build.allocation_invariants) == (
            family == "qwen35"
        )
    with pytest.raises(ValueError, match="router_statistics"):
        registry.resolve_objective("moe_balance", model=setup.model)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.parametrize("tied", [False, True])
def test_packed_weighted_loss_and_gradients_match_full_projection_reference(
    family, lora, tied
):
    _, setup, layouts = prepared(family, lora=lora, tied=tied)
    model = setup.model.create()
    reference_class = import_module(
        f"shadowspill_workloads.models.{family}.implementations.pytorch"
    ).Model
    reference = reference_class(model.config)
    if lora:
        reference = configure_lora(reference, rank=2, alpha=2, factor_dtype="float32")
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name.endswith("lora_b"):
                    parameter.normal_(std=0.02)
    reference.load_state_dict(model.state_dict())
    with bound(setup, layouts, datums()) as (binding, adapted):
        inputs, fields, controls = binding.arguments(adapted)
        actual = binding.objective(model, inputs, fields, controls)
        # The reference uses ordinary nonempty sequences; zero-capacity slots
        # have no rows and no state effects. Project each segment independently.
        lengths = [int(length) for length in inputs["sequence_lengths"] if length]
        logits = torch.cat(
            [reference(tokens) for tokens in inputs["tokens"].split(lengths, dim=1)],
            dim=1,
        ).float()
        losses = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), fields["target_tokens"].flatten(), reduction="none"
        ).reshape_as(fields["target_tokens"])
        expected = torch.where(fields["loss_mask"], losses * fields["weights"], 0).sum()
        torch.testing.assert_close(actual.loss, expected, rtol=5e-5, atol=5e-6)
        assert [n for n, p in model.named_parameters() if p.requires_grad] == [
            n for n, p in reference.named_parameters() if p.requires_grad
        ]
        actual_grad = torch.autograd.grad(
            actual.loss, [p for p in model.parameters() if p.requires_grad]
        )
        expected_grad = torch.autograd.grad(
            expected, [p for p in reference.parameters() if p.requires_grad]
        )
        torch.testing.assert_close(actual_grad, expected_grad, rtol=5e-4, atol=5e-6)


@pytest.mark.parametrize("family", FAMILIES)
def test_bfloat16_construction_preserves_catalog_storage_and_default_dtype(family):
    requested = settings(family, dtype="bfloat16")
    previous = torch.get_default_dtype()
    with torch.device("meta"):
        model = (
            load_registry(["shadowspill-workloads"])
            .resolve_model(family, settings=requested)
            .create()
        )
    assert torch.get_default_dtype() == previous
    assert model.embed.weight.dtype == torch.bfloat16
    assert model.rotary.cosine.dtype == torch.float32
    if family == "qwen35":
        assert model.blocks[0].mixer.A_log.dtype == torch.float32
        assert model.blocks[0].mixer.dt_bias.dtype == torch.float32


@pytest.mark.parametrize("settings_class", [Qwen3Settings, Qwen35Settings])
@pytest.mark.parametrize(
    "overrides",
    [
        {"implementation": "mlops_ep"},
        {"expert_options": {}},
        {"architecture": {"d_ff": 0}},
        {"architecture": {"tied_embeddings": 1}},
    ],
)
def test_dense_configuration_rejects_unsupported_settings(settings_class, overrides):
    with pytest.raises(ValueError):
        settings_class(**overrides)
