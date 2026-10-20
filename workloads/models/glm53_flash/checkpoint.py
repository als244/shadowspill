"""GLM-5.3-Flash naming atop the shared streaming checkpoint importer."""

from workloads.models.shared.glm.checkpoint import GLMCheckpoint as BaseCheckpoint

from .config import from_huggingface
from .implementations.mlops import Model, VisionConfig


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


class GLMCheckpoint(BaseCheckpoint):
    source_prefix = "model.language_model."
    config_from_metadata = staticmethod(from_huggingface)
    source_name = staticmethod(source_name)
    model_type = Model

    def vision_config(self, include_vision):
        return VisionConfig.from_huggingface(self.metadata) if include_vision else None
