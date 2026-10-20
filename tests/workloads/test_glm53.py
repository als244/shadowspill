"""GLM-5.3 architecture, packed causality, and gradients against HF attention."""

import inspect
from dataclasses import asdict

import pytest
import torch

pytest.importorskip("mlops")

from workloads.models.glm53.config import Config, from_huggingface
from workloads.models.glm53.implementations.mlops import Model
from workloads.models.glm53.implementations.mlops.attention import (
    SparseMLA,
    positions_for,
    rotate,
)
from workloads.models.glm53.quickstart import experiment


def test_normal_glm_architecture_and_lora_defaults_are_distinct_from_flash():
    c = Config()
    assert c.num_hidden_layers == 78
    assert c.indexer_types[:11] == (
        "full",
        "full",
        "full",
        "shared",
        "shared",
        "shared",
        "full",
        "shared",
        "shared",
        "shared",
        "full",
    )
    model = Model(Config.tiny(lora_rank=16), device="meta")
    assert not hasattr(model.layers[0], "attn_hc")
    assert model.layers[1].self_attn.indexer is None
    assert model.layers[0].self_attn.kv_a_proj_with_mqa.out_features == 128
    assert all(p.dtype == torch.bfloat16 for p in model.parameters() if p.requires_grad)
    defaults = inspect.signature(experiment).parameters
    assert defaults["lora_rank"].default == 16
    assert defaults["lora_dtype"].default == "bfloat16"
    assert defaults["grad_dtype"].default == "float32"


def test_checkpoint_metadata_rejects_other_architectures_and_rope_variants():
    with pytest.raises(ValueError, match="model_type"):
        from_huggingface({"model_type": "glm_moe_dsa_flash"})
    with pytest.raises(ValueError, match="RoPE"):
        from_huggingface({"rope_parameters": {"rope_type": "yarn"}})


def test_normal_recipe_keeps_bf16_casts_during_its_entire_lifetime():
    from torch._inductor import config as inductor_config

    recipe = experiment(
        device="cuda:0",
        tiny=True,
        sequence_length=16,
        sequences_per_step=1,
        sequences_per_microbatch=(1,),
    )
    with inductor_config.patch(emulate_precision_casts=False):
        with recipe["context"]():
            assert inductor_config.emulate_precision_casts
        assert not inductor_config.emulate_precision_casts


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("compiled", [False, True])
def test_normal_glm_attention_matches_hf_packed_outputs_and_input_gradients(compiled):
    pytest.importorskip("transformers.models.glm_moe_dsa.modeling_glm_moe_dsa")
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import (
        GlmMoeDsaConfig,
    )
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        GlmMoeDsaAttention,
        GlmMoeDsaRotaryEmbedding,
    )

    torch.manual_seed(771)
    c = Config.tiny()
    args = {k: v for k, v in asdict(c).items() if k not in ("dtype", "indexer_types")}
    args.update(
        n_routed_experts=c.num_local_experts,
        num_key_value_heads=c.num_attention_heads,
        indexer_types=list(c.indexer_types),
        rope_parameters={"rope_type": "default", "rope_theta": c.rope_theta},
    )
    reference_config = GlmMoeDsaConfig(**args)
    reference_config._attn_implementation = "eager"
    reference = GlmMoeDsaAttention(reference_config, 0).to(
        device="cuda", dtype=torch.bfloat16
    )
    model = SparseMLA(c, 0, "cuda")
    model.load_state_dict(reference.state_dict(), strict=True)
    rotary = GlmMoeDsaRotaryEmbedding(reference_config).to("cuda")
    x = torch.randn(
        2, 17, c.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    xr = x.detach().clone().requires_grad_()
    positions = torch.arange(17, device="cuda").expand(2, -1)
    embeddings = rotary(xr, positions)
    mask = torch.full((17, 17), -torch.inf, device="cuda").triu(1)[None, None]
    expected, _, reference_ids = reference(xr, embeddings, mask, position_ids=positions)
    boundaries = (0, 17, 34)
    fn = (
        torch.compile(model, fullgraph=True, options={"emulate_precision_casts": True})
        if compiled
        else model
    )
    actual, indices = fn(x.flatten(0, 1), boundaries, positions_for(boundaries, x))
    actual = actual.reshape_as(expected)
    for sequence in range(2):
        for row in range(17):
            valid = reference_ids[sequence, row]
            valid = valid[valid <= row].sort().values
            got = indices[sequence * 17 + row]
            got = (got[got >= 0] - sequence * 17).sort().values
            torch.testing.assert_close(got, valid, rtol=0, atol=0)
    dy = torch.randn_like(actual)
    grad = torch.autograd.grad(actual, x, dy)[0]
    reference_grad = torch.autograd.grad(expected, xr, dy)[0]
    for got, want in ((actual, expected), (grad, reference_grad)):
        error = float(
            ((got.float() - want.float()).norm() / want.float().norm()).detach()
        )
        assert error < 0.02, error


def test_interleaved_rope_uses_sequence_local_positions():
    x = torch.arange(4 * 64, dtype=torch.float32).reshape(4, 64) / 100
    result = rotate(x, torch.tensor([0, 1, 0, 1]), 8_000_000.0)
    torch.testing.assert_close(result[0], torch.cat((x[0, 0::2], x[0, 1::2])))
    torch.testing.assert_close(result[2], torch.cat((x[2, 0::2], x[2, 1::2])))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("storage", ["bf16", "fp8_block"])
def test_normal_model_compiles_with_all_lora_gradients_and_shared_indices(storage):
    if storage == "fp8_block" and torch.cuda.get_device_capability()[0] not in (9, 10):
        pytest.skip("FP8 expert LoRA uses SonicMoE on SM90/SM100")
    torch.manual_seed(92)
    model = Model(
        # Include every key/expert here so BF16 rounding cannot switch a
        # discrete route between the two complete forward passes. Selection
        # correctness, causal packing and subset keys are tested separately.
        Config.tiny(
            lora_rank=16,
            expert_storage=storage,
            index_topk=32,
            num_experts_per_tok=4,
        ),
        device="cuda",
    )
    with torch.no_grad():
        for name, value in model.named_parameters():
            if name.endswith(("lora_b", "lora_gate_up_b", "lora_down_b")):
                value.normal_(std=0.02)
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    tokens = torch.randint(model.config.vocab_size, (34,), device="cuda")
    targets = torch.randint(model.config.vocab_size, (34,), device="cuda")
    args = (tokens, targets, (0, 17, 34))
    expected = model.loss(*args, chunk_size=16)
    expected_gradients = torch.autograd.grad(expected, parameters)
    fn = torch.compile(
        model.loss,
        fullgraph=True,
        options={"emulate_precision_casts": True},
    )
    actual = fn(*args, chunk_size=16)
    gradients = torch.autograd.grad(actual, parameters)
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)
    for got, want in zip(gradients, expected_gradients, strict=True):
        assert torch.isfinite(got).all()
        error = (got.float() - want.float()).norm() / want.float().norm().clamp_min(
            1e-8
        )
        assert float(error) < 0.05, float(error)
