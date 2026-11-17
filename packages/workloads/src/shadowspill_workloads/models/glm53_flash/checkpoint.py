"""GLM-5.3-Flash naming atop the shared streaming checkpoint importer."""

from shadowspill_workloads.models.shared.glm.checkpoint import (
    GLMCheckpoint as BaseCheckpoint,
)

from .config import from_huggingface
from .implementations.mlops import Model, VisionConfig
from .weights import source_name


class GLMCheckpoint(BaseCheckpoint):
    source_prefix = "model.language_model."
    config_from_metadata = staticmethod(from_huggingface)
    source_name = staticmethod(source_name)
    model_type = Model

    def vision_config(self, include_vision):
        return VisionConfig.from_huggingface(self.metadata) if include_vision else None
