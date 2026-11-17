"""Checkpoint tensor naming; usable without a tensor framework."""

def source_name(name):
    name = name.replace(".mlp.router", ".mlp.gate.weight")
    name = name.replace(
        ".mlp.e_score_correction_bias", ".mlp.gate.e_score_correction_bias"
    )
    return name if name.startswith("lm_head.") else "model." + name

