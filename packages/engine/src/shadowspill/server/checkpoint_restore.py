"""Resolve owner-scoped checkpoints, without preparing or executing workers."""

import hashlib
import sys
from dataclasses import replace
from typing import BinaryIO

from shadowspill.training.session.checkpoint import ResolvedCheckpoint
from shadowspill.training.worker.frozen import validate_base
from shadowspill_client.protocol.checkpoints import (
    CHECKPOINT_MEDIA_TYPE,
    CheckpointManifest,
)
from shadowspill_client.protocol.codec import MAX_METADATA_BYTES, decode
from shadowspill_client.protocol.requests import (
    ArtifactRef,
    OperationRequest,
    identifier,
)
from shadowspill_client.protocol.training import ResolvedTraining

from .artifacts import ArtifactStore


def read_checkpoint_manifest(
    reference: ArtifactRef, stream: BinaryIO
) -> CheckpointManifest:
    """Decode only a bounded, checksum-verified portable generation manifest."""
    if (
        reference.media_type != CHECKPOINT_MEDIA_TYPE
        or reference.size_bytes > MAX_METADATA_BYTES
    ):
        raise ValueError("checkpoint is not a bounded generation manifest")
    body = stream.read(reference.size_bytes + 1)
    if (
        len(body) != reference.size_bytes
        or hashlib.sha256(body).hexdigest() != reference.sha256
    ):
        raise ValueError("checkpoint manifest differs from its immutable reference")
    value = decode(body)
    if not isinstance(value, CheckpointManifest):
        raise ValueError("artifact is not a checkpoint generation")
    return value


def require_checkpoint_dependencies(
    artifacts: ArtifactStore, manifest: CheckpointManifest, *, setup: bool = True
) -> None:
    """Registration requires all ranks; restore additionally requires its setup."""
    dependencies = [row.archive for row in manifest.ranks]
    if setup:
        dependencies[:0] = [manifest.info.configuration, manifest.info.plans]
        base = manifest.info.frozen_base
        if base is not None and base.artifact is not None:
            dependencies.append(base.artifact)
    for dependency in dependencies:
        if artifacts.get(dependency.id) != dependency:
            raise ValueError(f"checkpoint dependency is unavailable: {dependency.id}")
        with artifacts.open(dependency):
            pass


def validate_resume_optimizer(
    artifacts: ArtifactStore, source: ResolvedCheckpoint, target: ResolvedTraining
) -> None:
    """New setup may change defaults/objective, but cannot reinterpret saved state."""
    reference = source.manifest.info.configuration
    if reference.size_bytes > MAX_METADATA_BYTES:
        raise ValueError("checkpoint configuration exceeds its metadata bound")
    with artifacts.open(reference) as stream:
        body = stream.read(reference.size_bytes + 1)
    if (
        len(body) != reference.size_bytes
        or hashlib.sha256(body).hexdigest() != reference.sha256
    ):
        raise ValueError(
            "checkpoint configuration differs from its immutable reference"
        )
    saved = decode(body)
    if not isinstance(saved, ResolvedTraining):
        raise ValueError("checkpoint configuration is not resolved training settings")
    old, new = saved.optimizer, target.optimizer
    if (
        (old is None) != (new is None)
        or saved.definition_sources.get("optimizer")
        != target.definition_sources.get("optimizer")
        or (
            old is not None
            and new is not None
            and replace(old, defaults=new.defaults) != new
        )
    ):
        raise ValueError("checkpoint optimizer semantics require explicit conversion")


class CheckpointResolver:
    """Borrow storage under the prepared call's bounded metadata reservation."""

    def __init__(
        self,
        artifacts: ArtifactStore,
        *,
        session_id: str,
        world_size: int | None,
        configuration: ResolvedTraining,
    ):
        self.artifacts, self.session_id, self.world_size = (
            artifacts,
            session_id,
            world_size,
        )
        self.configuration = configuration

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
        return self.resolve(reference)

    def resolve(self, reference: ArtifactRef) -> ResolvedCheckpoint:
        """Creation supplies an immutable reference, without a run-local alias."""
        if self.artifacts.get(reference.id) != reference:
            raise ValueError("checkpoint is not in this owner's artifact storage")
        with self.artifacts.open(reference) as stream:
            value = read_checkpoint_manifest(reference, stream)
        info = value.info
        if info.byteorder != sys.byteorder:
            raise ValueError("checkpoint byte order requires explicit conversion")
        if self.world_size is not None and info.world_size != self.world_size:
            raise ValueError("checkpoint ownership requires explicit conversion")
        if info.frozen_state_id is not None and info.frozen_base is None:
            raise ValueError("this run has no declared frozen-base checkpoint identity")
        if info.frozen_base is not None:
            validate_base(info.frozen_base, self.configuration)
        require_checkpoint_dependencies(self.artifacts, value)
        resolved = ResolvedCheckpoint(reference, value)
        validate_resume_optimizer(self.artifacts, resolved, self.configuration)
        return resolved
