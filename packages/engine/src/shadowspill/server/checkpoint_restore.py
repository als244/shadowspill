"""Resolve owner-scoped checkpoints, without preparing or executing workers."""

import hashlib
import sys

from shadowspill.training.session.checkpoint import ResolvedCheckpoint
from shadowspill_client.protocol.checkpoints import (
    CHECKPOINT_MEDIA_TYPE,
    CheckpointManifest,
)
from shadowspill_client.protocol.codec import MAX_METADATA_BYTES, decode
from shadowspill_client.protocol.requests import OperationRequest, identifier

from .artifacts import ArtifactStore


class CheckpointResolver:
    """Borrow storage under the prepared call's bounded metadata reservation."""

    def __init__(self, artifacts: ArtifactStore, *, session_id: str, world_size: int):
        self.artifacts, self.session_id, self.world_size = (
            artifacts,
            session_id,
            world_size,
        )

    def __call__(self, request: OperationRequest) -> ResolvedCheckpoint:
        name = request.restore
        if request.operation != "restore" or name is None:
            raise ValueError("checkpoint resolution requires a restore request")
        try:
            identifier(name, "checkpoint ID")
        except ValueError:
            reference = None
        else:
            reference = self.artifacts.get(name)
        if reference is None:
            reference = self.artifacts.resolve_alias(
                self.session_id, request.identity.run_id, name
            )
        if reference is None:
            raise ValueError("checkpoint ID or run-local name does not exist")
        if (
            reference.media_type != CHECKPOINT_MEDIA_TYPE
            or reference.size_bytes > MAX_METADATA_BYTES
        ):
            raise ValueError("checkpoint is not a bounded generation manifest")
        with self.artifacts.open(reference) as stream:
            body = stream.read(reference.size_bytes + 1)
        if (
            len(body) != reference.size_bytes
            or hashlib.sha256(body).hexdigest() != reference.sha256
        ):
            raise ValueError("checkpoint manifest differs from its immutable reference")
        value = decode(body)
        if not isinstance(value, CheckpointManifest):
            raise ValueError("artifact is not a checkpoint generation")
        info = value.info
        if info.byteorder != sys.byteorder:
            raise ValueError("checkpoint byte order requires explicit conversion")
        if info.world_size != self.world_size:
            raise ValueError("checkpoint ownership requires explicit conversion")
        if info.frozen_state_id is not None:
            raise ValueError("this run has no declared frozen-base checkpoint identity")
        for dependency in (
            info.configuration,
            info.plans,
            *(rank.archive for rank in value.ranks),
        ):
            if self.artifacts.get(dependency.id) != dependency:
                raise ValueError(
                    f"checkpoint dependency is unavailable: {dependency.id}"
                )
        return ResolvedCheckpoint(reference, value)
