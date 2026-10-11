"""Translate the published architecture metadata without loading model weights."""

from dataclasses import fields

from .config import Config


def from_huggingface(config, **overrides):
    text = config.get("text_config", config)
    fields_by_name = {field.name for field in fields(Config)}
    values = {key: value for key, value in text.items() if key in fields_by_name}
    # HF's dtype is a string; arithmetic precision is an explicit caller option.
    values.pop("dtype", None)
    values["num_local_experts"] = text["n_routed_experts"]
    linear = text.get("linear_attn_config", {})
    for name, old in {
        "linear_num_heads": "num_heads",
        "linear_head_dim": "head_dim",
        "linear_conv_kernel_dim": "short_conv_kernel_size",
        "linear_lower_bound": "gate_lower_bound",
    }.items():
        if name not in values and old in linear:
            values[name] = linear[old]
    expected = {
        "mla_use_nope": True,
        "qk_rope_head_dim": 0,
        "mhc": True,
        "scoring_func": "sigmoid",
        "norm_topk_prob": True,
        "n_group": 1,
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "index_kpool_compress": True,
        "index_kpool_always_select_tail": True,
    }
    for key, value in expected.items():
        if text.get(key, value) != value:
            raise ValueError(
                f"Unsupported GLM architecture: {key}={text[key]!r}; expected {value!r}"
            )
    values.update(overrides)
    result = Config(**values)
    for key in ("layer_types", "mlp_layer_types"):
        if key in text and tuple(text[key]) != getattr(result, key):
            raise ValueError(
                f"Unsupported GLM {key}: checkpoint differs "
                "from configured architecture"
            )
    return result
