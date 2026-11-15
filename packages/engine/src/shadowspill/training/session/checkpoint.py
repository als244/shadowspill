"""Immutable restore input selected before durable request acceptance."""

from dataclasses import dataclass

from shadowspill_client.protocol.checkpoints import CheckpointManifest
from shadowspill_client.protocol.codec import MAX_METADATA_BYTES
from shadowspill_client.protocol.requests import ArtifactRef

# Covers the bounded global envelope, decoded record graph and retained ranks.
# Tensor descriptors remain in each worker's admitted artifact allowance.
CHECKPOINT_METADATA_BYTES = 4096 + 16 * MAX_METADATA_BYTES


@dataclass(frozen=True)
class ResolvedCheckpoint:
    reference: ArtifactRef
    manifest: CheckpointManifest
