"""GLM-5.3 naming atop the shared streaming checkpoint importer."""

from workloads.models.shared.glm.checkpoint import GLMCheckpoint as BaseCheckpoint

from .config import from_huggingface
from .implementations.mlops import Model


def source_name(name):
    name = name.replace(".mlp.router", ".mlp.gate.weight")
    name = name.replace(
        ".mlp.e_score_correction_bias", ".mlp.gate.e_score_correction_bias"
    )
    return name if name.startswith("lm_head.") else "model." + name


class GLMCheckpoint(BaseCheckpoint):
    source_prefix = "model."
    config_from_metadata = staticmethod(from_huggingface)
    source_name = staticmethod(source_name)
    model_type = Model
