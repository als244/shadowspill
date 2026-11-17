"""Checkpoint tensor naming; usable without a tensor framework."""

def source_name(name):
    if name.startswith("visual."):
        return "model." + name
    for short, original in (("attn_hc", "hc_attn"), ("ffn_hc", "hc_ffn")):
        for field in ("fn", "base", "scale"):
            name = name.replace(f".{short}.{field}", f".{original}_{field}")
    name = name.replace(".forget_gate.", ".")
    name = name.replace(".mlp.router", ".mlp.gate.weight")
    name = name.replace(
        ".mlp.e_score_correction_bias", ".mlp.gate.e_score_correction_bias"
    )
    return name if name.startswith("lm_head.") else "model.language_model." + name

