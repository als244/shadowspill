"""Checkpoint metadata and explicit local copies; no model or worker setup."""

from .generations import GenerationHandle
from .protocol.checkpoints import CHECKPOINT_MEDIA_TYPE, CheckpointManifest


class CheckpointHandle(GenerationHandle[CheckpointManifest]):
    """A saved checkpoint of a run's training state.

    `TrainingClient.save_checkpoint` returns one, as does
    `ArtifactTransfers.upload` for a checkpoint directory. It holds the
    weights, optimizer state, update count, random state and caller metadata,
    one file per worker. Pass its ``id`` to `TrainingClient.load_checkpoint`
    to load it into a run, or pass it as ``checkpoint=`` to
    `ServiceClient.create_training_client` to resume in a new run. Its methods
    are those of `GenerationHandle`.

    Attributes
    ----------
    reference : ArtifactRef
        The checkpoint manifest's artifact reference.
    """

    @property
    def _manifest_type(self) -> type[CheckpointManifest]:
        return CheckpointManifest

    @property
    def _media_type(self) -> str:
        return CHECKPOINT_MEDIA_TYPE
