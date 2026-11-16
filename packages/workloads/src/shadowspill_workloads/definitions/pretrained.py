"""Resolve supported Hub architectures through the public metadata interface."""

from dataclasses import asdict, replace
from importlib import import_module

from shadowspill.training.definitions import ModelReferenceDefinition
from shadowspill_client import Weights
from shadowspill_client._records import plain
from shadowspill_workloads.initialization import hub

from .source import definition_source

FAMILIES = {
    "llama": "llama3",
    "qwen3_moe": "qwen3moe",
    "qwen3_5_moe": "qwen35moe",
    "qwen3_5_moe_text": "qwen35moe",
    "olmoe": "olmoe",
}
# These settings select the stored tensors or their layout. Other supported
# overrides (e.g. rotary capacity/frequencies or top-k) do not reshape weights.
WEIGHT_GEOMETRY = {
    "n_layers",
    "d_model",
    "n_heads",
    "n_kv_heads",
    "head_dim",
    "d_ff",
    "vocab_size",
    "tied_embeddings",
    "n_experts",
    "d_ff_expert",
    "d_ff_shared",
    "full_attention_interval",
    "lin_k_heads",
    "lin_v_heads",
    "lin_k_head_dim",
    "lin_v_head_dim",
    "lin_conv_kernel",
    "attention_gate",
}


def architecture(metadata, overrides):
    text = metadata.get("text_config", metadata)
    if not isinstance(text, dict):
        raise ValueError("pretrained text_config must be a JSON object")
    kind = text.get("model_type", metadata.get("model_type"))
    if kind not in FAMILIES:
        raise ValueError(f"unsupported pretrained model type: {kind!r}")
    family = FAMILIES[kind]
    module = import_module(f"shadowspill_workloads.models.{family}.config")
    defaults = {}
    if "max_position_embeddings" in text:
        capacity = text["max_position_embeddings"]
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("pretrained maximum sequence length must be positive")
        defaults["max_seq_len"] = capacity
    original = module.from_huggingface(metadata, **defaults)
    _supported(text, original, family)
    values = asdict(original)
    for name in WEIGHT_GEOMETRY & overrides.keys():
        if name in values and values[name] != overrides[name]:
            raise ValueError(f"pretrained override changes checkpoint geometry: {name}")
    return family, asdict(replace(original, **overrides))


def _supported(text, config, family):
    if text.get("hidden_act", "silu") != "silu":
        raise ValueError("pretrained decoder requires the supported silu activation")
    if family == "llama3":
        return  # The Llama parser validates its supported biases and RoPE scaling.
    if text.get("tie_word_embeddings", False):
        raise ValueError("the supplied MoE decoders require untied embeddings")
    if text.get("rope_scaling") is not None:
        raise ValueError("the supplied MoE decoders do not implement rope_scaling")
    if text.get("mlp_only_layers") or text.get("decoder_sparse_step", 1) != 1:
        raise ValueError("the supplied MoE decoder requires every layer to be sparse")
    if family == "olmoe" and text.get("clip_qkv") is not None:
        raise ValueError("the supplied OLMoE decoder does not implement clip_qkv")
    layers = text.get("layer_types")
    layer_kind = getattr(config, "layer_kind", lambda index: "full")
    if layers is not None and (
        not isinstance(layers, list)
        or len(layers) != config.n_layers
        or any(
            kind != f"{layer_kind(index)}_attention"
            for index, kind in enumerate(layers)
        )
    ):
        raise ValueError("pretrained layer_types differ from the supported decoder")


def resolve(requested):
    weights = requested.initialize
    revision, metadata = hub.config(weights.reference, weights.revision)
    config = plain(requested.config)
    overrides = config.get("architecture", {})
    if not isinstance(overrides, dict):
        raise ValueError("pretrained architecture overrides must be a mapping")
    family, values = architecture(metadata, overrides)
    return replace(
        requested,
        definition=f"shadowspill/{family}@1",
        config={**config, "architecture": values},
        initialize=Weights.pretrained(weights.reference, revision=revision),
    )


def register(registry):
    registry.add_model_reference(
        ModelReferenceDefinition("shadowspill/huggingface@1", hub.accepts, resolve),
        source=definition_source(
            (
                "definitions/pretrained.py",
                "initialization/hub.py",
                "models/llama3/config.py",
                "models/olmoe/config.py",
                "models/qwen3moe/config.py",
                "models/qwen35moe/config.py",
                "models/shared/qwen_moe/config.py",
            ),
            providers=("huggingface_hub",),
        ),
    )
