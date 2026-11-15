"""Commit a verified rank generation through the existing artifact store.

This coordinator has no tensor serializer, training loop or worker launcher.
The ordered caller holds state quiescent, obtains exact rank descriptions before
streaming, and supplies installed writers that confirm the ordinary rank encoder's
returned manifest. ArtifactStore verifies their complete output bytes and commits
all ranks, the global manifest and a friendly name together.
"""

import hashlib
from collections.abc import Callable, Sequence

from shadowspill.orchestration.launch.publication import OutputSpec, OutputWriter
from shadowspill.orchestration.launch.staging import COPY_BYTES
from shadowspill.training.worker.checkpoint_records import RankSave, RankSaved
from shadowspill_client.protocol.checkpoints import (
    CHECKPOINT_MEDIA_TYPE,
    RANK_MEDIA_TYPE,
    CheckpointInfo,
    CheckpointManifest,
    CheckpointRank,
)
from shadowspill_client.protocol.codec import encode
from shadowspill_client.protocol.requests import ArtifactRef

from .artifact_group import StagedArtifacts
from .artifacts import MAX_GROUP_ARTIFACTS, ArtifactStore
from .records.artifact_index import ArtifactAlias


def checkpoint_id(info: CheckpointInfo) -> str:
    key = "\0".join((info.session_id, info.run_id, info.request_id))
    return "checkpoint." + hashlib.sha256(key.encode()).hexdigest()


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
    for dependency in (info.configuration, info.plans):
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
