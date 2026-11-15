"""Commit a verified rank generation through the existing artifact store.

This coordinator has no tensor serializer, training loop or worker launcher.
The ordered caller holds state quiescent, obtains exact rank descriptions before
streaming, and supplies installed writers that confirm the ordinary rank encoder's
returned manifest. ArtifactStore verifies their complete output bytes and commits
all ranks, the global manifest and a friendly name together.
"""

import hashlib
import sys
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from typing import BinaryIO

from shadowspill.orchestration.launch.publication import OutputSpec, OutputWriter
from shadowspill.orchestration.launch.staging import COPY_BYTES
from shadowspill.training.session.checkpoint import ResolvedCheckpoint
from shadowspill.training.session.prepared import PreparedCall
from shadowspill.training.session.state import Transition
from shadowspill.training.worker.checkpoint_records import (
    RankSave,
    RankSaved,
    checkpoint_id,
)
from shadowspill_client.protocol.checkpoints import (
    CHECKPOINT_MEDIA_TYPE,
    RANK_MEDIA_TYPE,
    CheckpointInfo,
    CheckpointManifest,
    CheckpointRank,
)
from shadowspill_client.protocol.codec import encode
from shadowspill_client.protocol.frozen import FrozenBase
from shadowspill_client.protocol.requests import ArtifactRef

from .artifact_group import StagedArtifacts
from .artifacts import MAX_GROUP_ARTIFACTS, ArtifactStore
from .records.artifact_index import ArtifactAlias


class RunCheckpoints:
    """Provide storage/provenance to an already initialized training run."""

    def __init__(
        self,
        artifacts: ArtifactStore,
        *,
        session_id: str,
        configuration: ArtifactRef,
        plans: ArtifactRef,
        world_size: int,
        frozen_base: FrozenBase | None = None,
    ) -> None:
        self.artifacts, self.session_id = artifacts, session_id
        self.configuration, self.plans, self.world_size = (
            configuration,
            plans,
            world_size,
        )
        self.frozen_base = frozen_base

    def restored(self, source: ResolvedCheckpoint) -> None:
        """Called only after every worker confirms the ordered restore."""
        self.frozen_base = source.manifest.info.frozen_base

    def open(self, reference: ArtifactRef) -> AbstractContextManager[BinaryIO]:
        return self.artifacts.open(reference)

    def describe(self, call: PreparedCall, transition: Transition) -> CheckpointInfo:
        request, settings = call.request, call.request.save
        if (
            settings is None
            or request.operation != "save"
            or transition.command != "save"
            or transition.request_id != request.identity.request_id
            or not transition.before.valid
            or transition.before.accumulation.open
        ):
            raise ValueError("checkpoint requires a valid completed-update boundary")
        return CheckpointInfo(
            self.session_id,
            request.identity.run_id,
            transition.before.history_id,
            request.identity.request_id,
            transition.before.completed_updates,
            self.world_size,
            self.configuration,
            self.plans,
            sys.byteorder,
            weights=settings.weights,
            metadata=settings.metadata,
            frozen_base=self.frozen_base,
            frozen_state_id=None
            if self.frozen_base is None
            else self.frozen_base.identity,
        )

    def stage(
        self,
        info: CheckpointInfo,
        ranks: Sequence[RankSave],
        write: Callable[[RankSave, OutputWriter], RankSaved],
        *,
        name: str | None,
    ) -> StagedArtifacts:
        return stage_checkpoint(self.artifacts, info, ranks, write, name=name)


def publish_checkpoint(
    artifacts: ArtifactStore,
    info: CheckpointInfo,
    ranks: Sequence[RankSave],
    write: Callable[[RankSave, OutputWriter], RankSaved],
    *,
    name: str | None = None,
) -> ArtifactRef:
    """Return a usable generation only after all state files persist together.

    This is for trusted installed rank writers, not unvalidated checkpoint uploads.
    The supplied description comes from the same worker boundary as its encoder.
    Rank archives retain the existing format and authoritative weight selection.
    """
    with stage_checkpoint(artifacts, info, ranks, write, name=name) as group:
        return group.commit()[-1]


def stage_checkpoint(
    artifacts: ArtifactStore,
    info: CheckpointInfo,
    ranks: Sequence[RankSave],
    write: Callable[[RankSave, OutputWriter], RankSaved],
    *,
    name: str | None = None,
) -> StagedArtifacts:
    """Return completed owned files for the existing dispatcher's publication phase.

    Keep mutation paused through commit or discard. A receipt failure closes this
    owner without making a checkpoint visible; publication never executes workers.
    """
    if len(ranks) + 1 > MAX_GROUP_ARTIFACTS:
        raise ValueError("checkpoint rank set exceeds the artifact publication bound")
    ranks = tuple(ranks)
    artifact_id = checkpoint_id(info)
    placeholders = tuple(
        rank.reference(f"{artifact_id}.rank.{rank.rank}", "0" * 64) for rank in ranks
    )
    # All digests have fixed width, so this is the exact metadata size, not a
    # guessed model envelope. Encode before invoking a rank or reserving disk.
    encoded_size = len(encode(CheckpointManifest(info, placeholders)))
    dependencies = [info.configuration, info.plans]
    if info.frozen_base is not None and info.frozen_base.artifact is not None:
        dependencies.append(info.frozen_base.artifact)
    for dependency in dependencies:
        if artifacts.get(dependency.id) != dependency:
            raise ValueError("checkpoint dependency is not committed for this owner")
    alias = (
        None
        if name is None
        else ArtifactAlias(info.session_id, info.run_id, name, artifact_id)
    )
    specs = [
        OutputSpec(row.archive.id, row.archive.size_bytes, RANK_MEDIA_TYPE)
        for row in placeholders
    ]
    specs.append(OutputSpec(artifact_id, encoded_size, CHECKPOINT_MEDIA_TYPE))
    by_id = {
        row.archive.id: rank for row, rank in zip(placeholders, ranks, strict=True)
    }
    completed: list[CheckpointRank] = []

    def produce(spec: OutputSpec, writer: OutputWriter) -> ArtifactRef:
        if spec.id != artifact_id:
            rank = by_id[spec.id]
            saved = write(rank, writer)
            rank.confirm(info, saved.manifest)
            row = rank.reference(spec.id, saved.sha256)
            completed.append(row)
            return row.archive
        body = encode(CheckpointManifest(info, completed))
        with memoryview(body) as data:
            offset = 0
            while offset < len(data):
                with data[offset : offset + COPY_BYTES] as part:
                    offset += writer.write(part)
        return ArtifactRef(
            artifact_id,
            hashlib.sha256(body).hexdigest(),
            len(body),
            CHECKPOINT_MEDIA_TYPE,
        )

    return artifacts.stage_group(specs, produce, alias=alias)
