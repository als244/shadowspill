"""Qwen 3: the published shapes' parameter counts, both implementations
against Hugging Face's on logits and every gradient, packing as independent
sequences, and the configurations the reader takes or refuses."""

from __future__ import annotations

from importlib import import_module

import pytest
import torch

from workloads.models.qwen3.config import PRESETS, Config, from_huggingface
from workloads.models.qwen3.implementations.pytorch import Model as Qwen3

#: The fields of Qwen3-1.7B's ``config.json`` the reader reads or checks.
QWEN3_1_7B = {
    "model_type": "qwen3",
    "num_hidden_layers": 28,
    "hidden_size": 2048,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 6144,
    "vocab_size": 151936,
    "rope_theta": 1000000,
    "rope_scaling": None,
    "rms_norm_eps": 1e-06,
    "tie_word_embeddings": True,
    "hidden_act": "silu",
    "attention_bias": False,
    "use_sliding_window": False,
}

#: Four heads of 16: the attention is twice the model's width.
TINY = Config(2, 32, 4, 2, 16, 64, 97, max_seq_len=16)


@pytest.mark.parametrize(
    "size,count",
    [("1.7b", 1_720_574_976), ("8b", 8_190_735_360), ("32b", 32_762_123_264)],
)
def test_the_published_shapes_count_their_parameters(size, count):
    with torch.device("meta"):
        model = Qwen3(PRESETS[size])
    assert sum(parameter.numel() for parameter in model.parameters()) == count


def _transformers_model(c):
    pytest.importorskip("transformers")
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    cfg = Qwen3Config(
        vocab_size=c.vocab_size,
        hidden_size=c.d_model,
        intermediate_size=c.d_ff,
        num_hidden_layers=c.n_layers,
        num_attention_heads=c.n_heads,
        num_key_value_heads=c.n_kv_heads,
        head_dim=c.head_dim,
        rms_norm_eps=c.norm_epsilon,
        rope_theta=c.rope_base,
        max_position_embeddings=c.max_seq_len,
        tie_word_embeddings=c.tied_embeddings,
        use_cache=False,
        attention_dropout=0.0,
    )
    cfg._attn_implementation = "eager"
    return Qwen3ForCausalLM(cfg)


def _copy_reference(model, reference):
    """Every parameter from the Hugging Face model, with what it came from."""
    checks = []

    def bind(target, source):
        with torch.no_grad():
            target.copy_(source)
        checks.append((target, source))

    bind(model.embed.weight, reference.model.embed_tokens.weight)
    if not model.config.tied_embeddings:
        bind(model.lm_head.weight, reference.lm_head.weight)
    bind(model.final_norm.weight, reference.model.norm.weight)
    for block, layer in zip(model.blocks, reference.model.layers, strict=True):
        bind(block.attn_norm.weight, layer.input_layernorm.weight)
        bind(block.ffn_norm.weight, layer.post_attention_layernorm.weight)
        for ours, theirs in (
            ("wq", "q_proj"),
            ("wk", "k_proj"),
            ("wv", "v_proj"),
            ("wo", "o_proj"),
            ("q_norm", "q_norm"),
            ("k_norm", "k_norm"),
        ):
            bind(
                getattr(block.attn, ours).weight,
                getattr(layer.self_attn, theirs).weight,
            )
        for ours, theirs in (
            ("w1", "gate_proj"),
            ("w3", "up_proj"),
            ("w2", "down_proj"),
        ):
            bind(getattr(block.mlp, ours).weight, getattr(layer.mlp, theirs).weight)
    return checks


@pytest.mark.parametrize("tied", [False, True], ids=["untied", "tied"])
@pytest.mark.parametrize("implementation", ["pytorch", "mlops"])
def test_logits_and_every_gradient_match_transformers(implementation, tied):
    torch.manual_seed(613)
    config = Config(**{**_fields(TINY), "tied_embeddings": tied})
    reference = _transformers_model(config)
    model = import_module(
        f"workloads.models.qwen3.implementations.{implementation}"
    ).Model(config)
    checks = _copy_reference(model, reference)
    assert len(checks) == len(list(model.parameters()))
    tokens = torch.randint(config.vocab_size, (1, 7))
    expected = reference(tokens, use_cache=False).logits
    actual = model(tokens)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    # An arbitrary upstream gradient exercises every projection without
    # conflating model math with HF's shifted-label loss convention.
    upstream = torch.randn_like(actual)
    (actual * upstream).sum().backward()
    (expected * upstream).sum().backward()
    for target, source in checks:
        torch.testing.assert_close(target.grad, source.grad, atol=8e-5, rtol=8e-4)


def _fields(config):
    return {name: getattr(config, name) for name in config.__slots__}


def test_packed_matches_independent_sequences():
    torch.manual_seed(5)
    model = Qwen3(Config(1, 32, 4, 2, 16, 64, 97, max_seq_len=16)).eval()
    first = torch.randint(0, 97, (1, 3))
    second = torch.randint(0, 97, (1, 5))
    expected = torch.cat((model(first), model(second)), dim=1)
    actual = model(torch.cat((first, second), dim=1), (3, 5))
    torch.testing.assert_close(actual, expected)


def test_the_reader_takes_a_published_config_in_either_rotary_form():
    assert from_huggingface(QWEN3_1_7B) == PRESETS["1.7b"]
    newer = {
        key: value
        for key, value in QWEN3_1_7B.items()
        if key not in ("rope_theta", "rope_scaling")
    }
    newer["rope_parameters"] = {"rope_type": "default", "rope_theta": 1000000}
    assert from_huggingface(newer) == PRESETS["1.7b"]
    assert from_huggingface(QWEN3_1_7B, max_seq_len=4096).max_seq_len == 4096


@pytest.mark.parametrize(
    "change",
    [
        {"model_type": "qwen3_moe"},
        {"attention_bias": True},
        {"use_sliding_window": True},
        {"hidden_act": "gelu"},
        {"rope_scaling": {"rope_type": "yarn", "factor": 4.0}},
        {"rope_parameters": {"rope_type": "yarn", "rope_theta": 1000000}},
    ],
    ids=["mixture", "bias", "sliding", "activation", "scaling", "parameters"],
)
def test_the_reader_refuses_what_the_definition_does_not_reproduce(change):
    with pytest.raises(ValueError):
        from_huggingface({**QWEN3_1_7B, **change})


def test_unknown_fields_and_bad_head_geometry_are_refused():
    with pytest.raises(ValueError, match="unknown"):
        from_huggingface(QWEN3_1_7B, layers=2)
    with pytest.raises(ValueError, match="divisible"):
        Config(1, 32, 3, 2, 16, 64, 97)
    with pytest.raises(ValueError, match="even"):
        Config(1, 32, 4, 2, 15, 64, 97)
