"""Inference exports retain the same explicit, bounded artifact transfer API."""

from .generations import GenerationHandle
from .protocol.checkpoints import WEIGHTS_MEDIA_TYPE, WeightsManifest


class WeightsHandle(GenerationHandle[WeightsManifest]):
    @property
    def _manifest_type(self) -> type[WeightsManifest]:
        return WeightsManifest

    @property
    def _media_type(self) -> str:
        return WEIGHTS_MEDIA_TYPE
