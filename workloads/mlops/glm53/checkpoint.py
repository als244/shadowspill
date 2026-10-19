"""GLM-5.3 naming atop the shared streaming FP8/checkpoint importer."""

from ..glm53_flash.checkpoint import GLMCheckpoint as BaseCheckpoint
from .hf_config import from_huggingface
from .model import LanguageModel


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
    model_type = LanguageModel

    def build_model(self, *, include_vision=False, **kwargs):
        if include_vision:
            raise ValueError("GLM-5.3 is a text-only workload")
        return super().build_model(**kwargs)
