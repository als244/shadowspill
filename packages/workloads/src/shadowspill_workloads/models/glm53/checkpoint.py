"""GLM-5.3 naming atop the shared streaming checkpoint importer."""

from shadowspill_workloads.models.shared.glm.checkpoint import (
    GLMCheckpoint as BaseCheckpoint,
)

from .config import from_huggingface
from .implementations.mlops import Model
from .weights import source_name


class GLMCheckpoint(BaseCheckpoint):
    source_prefix = "model."
    config_from_metadata = staticmethod(from_huggingface)
    source_name = staticmethod(source_name)
    model_type = Model
