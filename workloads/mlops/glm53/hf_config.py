"""Validate and translate GLM-5.3 metadata without reading weight payloads."""

from dataclasses import fields

from .config import Config


def from_huggingface(metadata, **overrides):
    expected = {
        "model_type": "glm_moe_dsa",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "hidden_act": "silu",
        "moe_layer_freq": 1,
        "scoring_func": "sigmoid",
        "norm_topk_prob": True,
        "n_group": 1,
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "rope_interleave": True,
        "indexer_rope_interleave": True,
        "tie_word_embeddings": False,
    }
    for key, value in expected.items():
        if metadata.get(key, value) != value:
            raise ValueError(f"Unsupported GLM-5.3 {key}: {metadata[key]!r}")
    rope = metadata.get("rope_parameters", {})
    if rope.get("rope_type", "default") != "default":
        raise ValueError("Only the published default GLM-5.3 RoPE is supported")
    accepted = {f.name for f in fields(Config)} - {"dtype"}
    values = {k: v for k, v in metadata.items() if k in accepted}
    values["num_local_experts"] = metadata["n_routed_experts"]
    values["rope_theta"] = rope.get("rope_theta", 8_000_000.0)
    values.update(overrides)
    config = Config(**values)
    if (
        "mlp_layer_types" in metadata
        and tuple(metadata["mlp_layer_types"]) != config.mlp_layer_types
    ):
        raise ValueError(
            "Checkpoint MLP layer types disagree with configured architecture"
        )
    return config
