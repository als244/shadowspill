"""Checkpoint metadata and explicit local copies; no model or worker setup."""

from .generations import GenerationHandle
from .protocol.checkpoints import CHECKPOINT_MEDIA_TYPE, CheckpointManifest


class CheckpointHandle(GenerationHandle[CheckpointManifest]):
    @property
    def _manifest_type(self) -> type[CheckpointManifest]:
        return CheckpointManifest

    @property
    def _media_type(self) -> str:
        return CHECKPOINT_MEDIA_TYPE
