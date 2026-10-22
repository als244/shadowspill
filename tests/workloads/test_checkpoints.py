"""Every family reads a Hugging Face checkpoint of its own layout: a tiny
model's parameters, written as the checkpoint would hold them, come back
equal through the reader."""

import json

import pytest
import torch
from safetensors.torch import save_file

from workloads.models import catalog
from workloads.models.shared.checkpoint import copy_into

READERS = {
    "llama3": ("workloads.models.llama3.checkpoint", "Llama3Checkpoint"),
    "qwen3": ("workloads.models.qwen3.checkpoint", "Qwen3Checkpoint"),
    "qwen35": ("workloads.models.qwen35.checkpoint", "Qwen35Checkpoint"),
    "olmoe": ("workloads.models.olmoe.checkpoint", "OLMoECheckpoint"),
    "qwen3moe": ("workloads.models.qwen3moe.checkpoint", "Qwen3MoECheckpoint"),
    "qwen35moe": ("workloads.models.qwen35moe.checkpoint", "Qwen35MoECheckpoint"),
}


def reader(family):
    import importlib

    module, name = READERS[family]
    return getattr(importlib.import_module(module), name)


def interleave_heads(weight, heads):
    """The inverse of the reader's de-interleaving: all queries then all
    gates, to query and gate per head."""
    rows, width = weight.shape
    return (
        weight.view(2, heads, rows // (2 * heads), width)
        .transpose(0, 1)
        .reshape(rows, width)
    )


def huggingface_state(family, model, config):
    """The tiny model's parameters under the names and layouts of the
    family's Hugging Face checkpoint, written independently of the reader."""

    state = {}
    parameters = dict(model.named_parameters())
    prefix = "model.language_model." if family in ("qwen35", "qwen35moe") else "model."
    state[prefix + "embed_tokens.weight"] = parameters["embed.weight"]
    state[prefix + "norm.weight"] = parameters["final_norm.weight"]
    if not getattr(config, "tied_embeddings", False):
        state["lm_head.weight"] = parameters["lm_head.weight"]
    for layer in range(config.n_layers):
        at = f"{prefix}layers.{layer}."
        ours = {
            key[len(f"blocks.{layer}.") :]: value
            for key, value in parameters.items()
            if key.startswith(f"blocks.{layer}.")
        }
        state[at + "input_layernorm.weight"] = ours["attn_norm.weight"]
        state[at + "post_attention_layernorm.weight"] = ours["ffn_norm.weight"]
        mixer = "attn" if family in ("llama3", "qwen3", "olmoe") else "mixer"
        kind = config.layer_kind(layer) if hasattr(config, "layer_kind") else "full"
        if kind == "full":
            query = ours[f"{mixer}.wq.weight"]
            if getattr(config, "attention_gate", family == "qwen35"):
                query = interleave_heads(query, config.n_heads)
            state[at + "self_attn.q_proj.weight"] = query
            for ours_name, theirs in (
                ("wk", "k_proj"),
                ("wv", "v_proj"),
                ("wo", "o_proj"),
            ):
                state[at + f"self_attn.{theirs}.weight"] = ours[
                    f"{mixer}.{ours_name}.weight"
                ]
            for norm in ("q_norm", "k_norm"):
                if f"{mixer}.{norm}.weight" in ours:
                    state[at + f"self_attn.{norm}.weight"] = ours[
                        f"{mixer}.{norm}.weight"
                    ]
        else:
            qkvz = ours["mixer.w_qkvz.weight"]
            state[at + "linear_attn.in_proj_qkv.weight"] = qkvz[
                : config.convolution_width
            ]
            state[at + "linear_attn.in_proj_z.weight"] = qkvz[
                config.convolution_width :
            ]
            ba = ours["mixer.w_ba.weight"]
            state[at + "linear_attn.in_proj_b.weight"] = ba[: config.lin_v_heads]
            state[at + "linear_attn.in_proj_a.weight"] = ba[config.lin_v_heads :]
            state[at + "linear_attn.conv1d.weight"] = ours["mixer.conv.weight"]
            state[at + "linear_attn.A_log"] = ours["mixer.A_log"]
            state[at + "linear_attn.dt_bias"] = ours["mixer.dt_bias"]
            state[at + "linear_attn.norm.weight"] = ours["mixer.lin_norm.weight"]
            state[at + "linear_attn.out_proj.weight"] = ours["mixer.w_out.weight"]
        if "mlp.w1.weight" in ours:
            state[at + "mlp.gate_proj.weight"] = ours["mlp.w1.weight"]
            state[at + "mlp.up_proj.weight"] = ours["mlp.w3.weight"]
            state[at + "mlp.down_proj.weight"] = ours["mlp.w2.weight"]
        if "moe.router.weight" in ours:
            state[at + "mlp.gate.weight"] = ours["moe.router.weight"]
            w13 = ours.get("moe.w13", ours.get("moe.w13_experts"))
            w2 = ours.get("moe.w2", ours.get("moe.w2_experts"))
            if family == "qwen35moe":
                state[at + "mlp.experts.gate_up_proj"] = w13
                state[at + "mlp.experts.down_proj"] = w2
                state[at + "mlp.shared_expert.gate_proj.weight"] = ours[
                    "moe.shared.gate_up.weight"
                ][: config.d_ff_shared]
                state[at + "mlp.shared_expert.up_proj.weight"] = ours[
                    "moe.shared.gate_up.weight"
                ][config.d_ff_shared :]
                state[at + "mlp.shared_expert.down_proj.weight"] = ours[
                    "moe.shared.down.weight"
                ]
                state[at + "mlp.shared_expert_gate.weight"] = ours[
                    "moe.shared.gate.weight"
                ]
            else:
                half = config.d_ff_expert
                for index in range(config.n_experts):
                    state[at + f"mlp.experts.{index}.gate_proj.weight"] = w13[
                        index, :half
                    ]
                    state[at + f"mlp.experts.{index}.up_proj.weight"] = w13[
                        index, half:
                    ]
                    state[at + f"mlp.experts.{index}.down_proj.weight"] = w2[index]
    return {key: value.detach().clone().contiguous() for key, value in state.items()}


def metadata(family, config):
    """A ``config.json`` the family's reader accepts for the tiny shape."""

    common = {
        "num_hidden_layers": config.n_layers,
        "hidden_size": config.d_model,
        "num_attention_heads": config.n_heads,
        "num_key_value_heads": config.n_kv_heads,
        "vocab_size": config.vocab_size,
        "attention_bias": False,
    }
    if family == "llama3":
        return {
            **common,
            "model_type": "llama",
            "intermediate_size": config.d_ff,
            "rope_theta": config.rope_base,
            "tie_word_embeddings": config.tied_embeddings,
            "rms_norm_eps": config.norm_epsilon,
        }
    if family == "qwen3":
        return {
            **common,
            "model_type": "qwen3",
            "head_dim": config.head_dim,
            "intermediate_size": config.d_ff,
            "rope_theta": config.rope_base,
            "tie_word_embeddings": config.tied_embeddings,
            "rms_norm_eps": config.norm_epsilon,
        }
    if family == "olmoe":
        return {
            **common,
            "model_type": "olmoe",
            "intermediate_size": config.d_ff_expert,
            "num_experts": config.n_experts,
            "num_experts_per_tok": config.top_k,
            "rope_theta": config.rope_base,
            "norm_topk_prob": False,
        }
    linear = {
        "linear_num_key_heads": config.lin_k_heads,
        "linear_num_value_heads": config.lin_v_heads,
        "linear_key_head_dim": config.lin_k_head_dim,
        "linear_value_head_dim": config.lin_v_head_dim,
        "linear_conv_kernel_dim": config.lin_conv_kernel,
    }
    rope = {
        "rope_type": "default",
        "rope_theta": config.rope_base,
        "partial_rotary_factor": config.partial_rotary_factor,
    }
    if family == "qwen35":
        text = {
            **common,
            "model_type": "qwen3_5_text",
            "head_dim": config.head_dim,
            "full_attention_interval": config.full_attention_interval,
            "intermediate_size": config.d_ff,
            "rope_parameters": rope,
            "rms_norm_eps": config.norm_epsilon,
            "tie_word_embeddings": config.tied_embeddings,
            **linear,
        }
        return {"model_type": "qwen3_5", "text_config": text}
    if family == "qwen3moe":
        return {
            **common,
            "model_type": "qwen3_moe",
            "head_dim": config.head_dim,
            "moe_intermediate_size": config.d_ff_expert,
            "num_experts": config.n_experts,
            "num_experts_per_tok": config.top_k,
            "rope_theta": config.rope_base,
            "rms_norm_eps": config.norm_epsilon,
            "router_aux_loss_coef": config.router_aux_loss_coef,
            "norm_topk_prob": True,
        }
    text = {
        **common,
        "model_type": "qwen3_5_moe_text",
        "head_dim": config.head_dim,
        "moe_intermediate_size": config.d_ff_expert,
        "shared_expert_intermediate_size": config.d_ff_shared,
        "num_experts": config.n_experts,
        "num_experts_per_tok": config.top_k,
        "full_attention_interval": config.full_attention_interval,
        "rope_parameters": rope,
        "rms_norm_eps": config.norm_epsilon,
        "router_aux_loss_coef": config.router_aux_loss_coef,
        **linear,
    }
    return {"model_type": "qwen3_5_moe", "text_config": text}


def built(name, **changes):
    """A tiny model built on the CPU as the catalog builds on meta, its ties
    intact (``to_empty`` would separate a tied head from its embedding)."""
    entry = catalog.entry(name)
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cpu"):
            return entry.model_class("mlops")(config=entry.config(**changes))
    finally:
        torch.set_default_dtype(previous)


def materialized(name, **changes):
    model = built(name, **changes)
    torch.manual_seed(0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(
                torch.randn_like(parameter, dtype=torch.float32).to(parameter.dtype)
            )
    return model


#: What the tiny shapes leave at their defaults but every published
#: checkpoint of the family has, which its reader therefore sets.
PUBLISHED = {"qwen35": {"zero_centered_norm": True}}


@pytest.mark.parametrize("family", sorted(READERS))
def test_a_family_reads_its_own_layout_back(tmp_path, family):
    name = f"{family}-tiny"
    entry = catalog.entry(name)
    changes = PUBLISHED.get(family, {})
    source = materialized(name, **changes)
    config = entry.config(**changes)
    state = huggingface_state(family, source, config)
    directory = tmp_path / "checkpoint"
    directory.mkdir()
    save_file(state, directory / "model.safetensors")
    (directory / "config.json").write_text(json.dumps(metadata(family, config)))
    checkpoint = reader(family)(directory, max_seq_len=config.max_seq_len)
    assert checkpoint.config == config
    target = built(name, **changes)
    with torch.no_grad():
        for parameter in target.parameters():
            parameter.zero_()
    with checkpoint:
        report = checkpoint.load_into(target)
    assert report["unused"] == []
    expected = dict(source.named_parameters())
    for parameter_name, parameter in target.named_parameters():
        assert torch.equal(parameter, expected[parameter_name]), parameter_name
    assert set(report["written"]) <= set(expected)


def test_qwen_lora_initializer_loads_base_and_initializes_only_adapters_and_buffers(
    tmp_path,
    monkeypatch,
):
    from workloads.lora import configure_lora
    from workloads.models.shared import checkpoint as checkpoint_module

    opened = []
    original = checkpoint_module.SafeTensors

    def track(directory):
        result = original(directory)
        opened.append(result)
        return result

    monkeypatch.setattr(checkpoint_module, "SafeTensors", track)

    family, name = "qwen35moe", "qwen35moe-tiny"
    source = materialized(name)
    config = source.config
    save_file(huggingface_state(family, source, config), tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text(json.dumps(metadata(family, config)))
    target = configure_lora(built(name), rank=4, alpha=8)
    with torch.no_grad():
        for value in (*target.parameters(), *target.buffers()):
            value.fill_(float("nan"))
    checkpoint = reader(family)(tmp_path, max_seq_len=config.max_seq_len)
    report = checkpoint.initialize(target)
    assert opened[-1]._closed and not opened[-1]._handles
    # The initializer is reusable and leaves no mapped checkpoint files behind.
    report = checkpoint.initialize(target)
    assert opened[-1]._closed and not opened[-1]._handles
    assert not checkpoint.reader._handles
    assert report["unused"] == [] and report["initialized_adapters"]
    base = dict(source.named_parameters())
    for name, parameter in target.named_parameters():
        assert torch.isfinite(parameter).all(), name
        if name in base:
            torch.testing.assert_close(parameter, base[name], rtol=0, atol=0)
        elif name.endswith("_b"):
            assert torch.count_nonzero(parameter) == 0
        else:
            assert torch.count_nonzero(parameter) > 0
    for name, buffer in target.named_buffers():
        torch.testing.assert_close(buffer, dict(source.named_buffers())[name])


@pytest.mark.parametrize("family", ["llama3", "qwen3"])
def test_a_tied_head_is_written_once_from_the_embedding(tmp_path, family):
    name = f"{family}-tiny"
    entry = catalog.entry(name)
    config = entry.config(tied_embeddings=True)
    source = materialized(name, tied_embeddings=True)
    state = huggingface_state(family, source, config)
    assert "lm_head.weight" not in state
    directory = tmp_path / "checkpoint"
    directory.mkdir()
    save_file(state, directory / "model.safetensors")
    (directory / "config.json").write_text(json.dumps(metadata(family, config)))
    target = built(name, tied_embeddings=True)
    with reader(family)(directory, max_seq_len=config.max_seq_len) as checkpoint:
        report = checkpoint.load_into(target)
    assert target.lm_head.weight is target.embed.weight
    assert torch.equal(target.embed.weight, source.embed.weight)
    assert "lm_head.weight" not in report["written"] and report["unused"] == []


def test_a_tied_checkpoint_may_store_its_head_only_as_the_embedding(tmp_path):
    # Qwen 3's tied checkpoints store lm_head.weight beside the embedding.
    name = "qwen3-tiny"
    config = catalog.entry(name).config(tied_embeddings=True)
    source = materialized(name, tied_embeddings=True)
    state = huggingface_state("qwen3", source, config)
    state["lm_head.weight"] = state["model.embed_tokens.weight"].clone()
    directory = tmp_path / "checkpoint"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(metadata("qwen3", config)))
    save_file(state, directory / "model.safetensors")
    target = built(name, tied_embeddings=True)
    with reader("qwen3")(directory, max_seq_len=config.max_seq_len) as checkpoint:
        report = checkpoint.load_into(target)
    assert report["unused"] == []
    assert target.lm_head.weight is target.embed.weight
    assert torch.equal(target.embed.weight, source.embed.weight)
    state["lm_head.weight"] = state["lm_head.weight"] + 1
    save_file(state, directory / "model.safetensors")
    with (
        reader("qwen3")(directory, max_seq_len=config.max_seq_len) as checkpoint,
        pytest.raises(ValueError, match="differs"),
    ):
        checkpoint.load_into(built(name, tied_embeddings=True))


def test_copy_into_refuses_a_shape_that_does_not_fit():
    parameter = torch.nn.Parameter(torch.zeros(4, 2))
    copy_into(parameter, torch.ones(4, 2))
    assert torch.equal(parameter, torch.ones(4, 2))
    with pytest.raises(ValueError):
        copy_into(parameter, torch.ones(2, 4))


def test_a_name_the_layout_does_not_know_is_refused(tmp_path):
    config = catalog.entry("llama3-tiny").config()
    directory = tmp_path / "checkpoint"
    directory.mkdir()
    save_file({"x": torch.zeros(1)}, directory / "model.safetensors")
    (directory / "config.json").write_text(json.dumps(metadata("llama3", config)))
    with (
        reader("llama3")(directory) as checkpoint,
        pytest.raises(ValueError, match="no checkpoint source"),
    ):
        checkpoint.sources("blocks.0.attn.wz.weight", config)
