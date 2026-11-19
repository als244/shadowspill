"""Inference exports retain the same explicit, bounded artifact transfer API."""

from .generations import GenerationHandle
from .protocol.checkpoints import WEIGHTS_MEDIA_TYPE, WeightsManifest


class WeightsHandle(GenerationHandle[WeightsManifest]):
    """Exported model weights and persistent buffers, without optimizer state.

    `TrainingClient.export_weights` returns one, as does
    `ArtifactTransfers.upload` for a weights directory. Start a new run from
    them with ``Weights.artifact(weights.id)`` as the model's initialization.
    They cannot be loaded as a checkpoint. Its methods are those of
    `GenerationHandle`.

    Attributes
    ----------
    reference : ArtifactRef
        The export manifest's artifact reference.
    """

    @property
    def _manifest_type(self) -> type[WeightsManifest]:
        return WeightsManifest

    @property
    def _media_type(self) -> str:
        return WEIGHTS_MEDIA_TYPE
