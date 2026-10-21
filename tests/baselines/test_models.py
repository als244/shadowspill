"""The catalog model an engine builds, its objective and its useful work."""

import argparse
import functools

import pytest
import torch

from baselines.common import models
from baselines.common.geometry import Geometry
from baselines.common.models import (
    ModelRequest,
    full_attention_layers,
    linear_work,
    useful_flops_per_token,
)

GEOMETRY = Geometry(1024, 8192, 2048)


def test_a_request_names_a_catalog_model_an_implementation_and_a_mode():
    request = ModelRequest("llama3-tiny", "pytorch")
    assert request.entry.family == "llama3"
    assert request.as_dict() == {
        "model": "llama3-tiny",
        "implementation": "pytorch",
        "mode": "full",
        "lora": {},
        "config": {},
    }
    with pytest.raises(ValueError, match="implementation"):
        ModelRequest("llama3-tiny", "jax")
    with pytest.raises(ValueError, match="mode"):
        ModelRequest("llama3-tiny", "pytorch", "partial")
    with pytest.raises(ValueError, match="factory"):
        ModelRequest("glm53", "mlops")
    with pytest.raises(ValueError, match="LoRA"):
        ModelRequest("llama3-tiny", "pytorch", "lora", {"depth": 2})
    with pytest.raises(ValueError, match="mode=lora"):
        ModelRequest("llama3-tiny", "pytorch", "full", {"rank": 4})
    with pytest.raises(ValueError):
        ModelRequest("llama3-tiny", "pytorch", config={"n_towers": 2})


def test_the_model_is_built_on_meta_with_lora_applied_in_that_mode():
    full = ModelRequest("llama3-tiny", "pytorch")
    model = full.build(dtype="bfloat16")
    assert all(p.device.type == "meta" for p in model.parameters())
    assert model.embed.weight.dtype == torch.bfloat16
    summary = full.summary(model)
    assert summary["parameters"] == sum(p.numel() for p in model.parameters())
    assert summary["trainable_parameters"] == summary["parameters"]
    assert summary["tied_embeddings"] is False
    assert summary["mixture"] is False

    adapted = ModelRequest("llama3-tiny", "pytorch", "lora", {"rank": 4})
    model = adapted.build(dtype="bfloat16")
    trained = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trained and all("lora_" in name for name in trained)
    summary = adapted.summary(model)
    assert 0 < summary["trainable_parameters"] < summary["parameters"]
    assert summary["lora"] == {"rank": 4}

    tied = ModelRequest("llama3-tiny", "pytorch", config={"tied_embeddings": True})
    assert tied.summary(tied.build(dtype="bfloat16"))["tied_embeddings"] is True


def test_the_spec_and_objective_follow_the_rank_geometry():
    request = ModelRequest("llama3-tiny", "pytorch")
    spec = request.spec(GEOMETRY)
    assert (spec.model, spec.implementation) == ("llama3-tiny", "pytorch")
    assert spec.sequence_length == 1024
    assert spec.sequences_per_microbatch == 2
    assert spec.accumulation_count == 4
    objective = request.objective(GEOMETRY)
    assert isinstance(objective, functools.partial)
    assert objective.args[0] == spec


def test_useful_work_counts_the_matrices_a_token_multiplies():
    model = ModelRequest("llama3-tiny", "pytorch").build(dtype="bfloat16")
    config = model.config
    heads = 2 * config.n_heads + 2 * config.n_kv_heads
    attention = config.d_model * heads * config.head_dim
    mlp = 3 * config.d_model * config.d_ff
    head = config.vocab_size * config.d_model
    expected = config.n_layers * (attention + mlp) + head
    assert linear_work(model) == {"frozen": 0.0, "trainable": float(expected)}
    assert full_attention_layers(model) == config.n_layers
    width = config.n_heads * config.head_dim
    scores = 12 * 1024 * width * config.n_layers
    assert useful_flops_per_token(model, 1024) == 6 * expected + scores
    with pytest.raises(ValueError):
        useful_flops_per_token(model, 0)


def test_lora_work_counts_the_frozen_base_and_the_trained_factors():
    adapted = ModelRequest("llama3-tiny", "pytorch", "lora", {"rank": 4})
    model = adapted.build(dtype="bfloat16")
    work = linear_work(model)
    factors = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert work["trainable"] == float(factors)
    base = ModelRequest("llama3-tiny", "pytorch").build(dtype="bfloat16")
    assert work["frozen"] == linear_work(base)["trainable"]


def test_a_routed_bank_counts_the_experts_a_token_visits():
    model = ModelRequest("olmoe-tiny", "pytorch").build(dtype="bfloat16")
    config = model.config
    embedding = model.embed.weight
    dense = sum(
        p.numel() for p in model.parameters() if p.ndim == 2 and p is not embedding
    )
    banks = sum(p.numel() for p in model.parameters() if p.ndim == 3)
    visited = banks * config.top_k / config.n_experts
    assert linear_work(model)["trainable"] == dense + visited
    assert full_attention_layers(model) == config.n_layers


def test_linear_attention_blocks_are_not_full_attention():
    model = ModelRequest("qwen35-tiny", "pytorch").build(dtype="bfloat16")
    kinds = [block.kind for block in model.blocks]
    assert full_attention_layers(model) == kinds.count("full") < len(kinds)


def parse(*argv):
    parser = argparse.ArgumentParser()
    models.add_arguments(parser)
    return parser.parse_args(argv)


def test_flags_build_a_request():
    parsed = parse(
        "--model",
        "llama3-tiny",
        "--implementation",
        "pytorch",
        "--mode",
        "lora",
        "--lora",
        '{"rank": 4}',
        "--model-config",
        '{"n_layers": 1}',
    )
    expected = ModelRequest(
        "llama3-tiny", "pytorch", "lora", {"rank": 4}, {"n_layers": 1}
    )
    assert models.from_arguments(parsed) == expected
    default = models.from_arguments(parse("--model", "llama3-tiny"))
    assert default.implementation == "mlops"
    with pytest.raises(SystemExit):
        parse("--model", "llama2-7b")
