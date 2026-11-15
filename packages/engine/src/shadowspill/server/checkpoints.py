"""Commit a verified rank generation through the existing artifact store.

This coordinator has no tensor serializer, training loop or worker launcher.
The ordered caller holds state quiescent, obtains exact rank descriptions before
streaming, and supplies installed writers that confirm the ordinary rank encoder's
returned manifest. ArtifactStore verifies their complete output bytes and commits
all ranks, the global manifest and a friendly name together.
"""

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from shadowspill.orchestration.launch.publication import OutputSpec, OutputWriter
from shadowspill.orchestration.launch.staging import COPY_BYTES
from shadowspill_client._records import integer
from shadowspill_client.protocol.checkpoints import (
    CHECKPOINT_MEDIA_TYPE,
    RANK_MEDIA_TYPE,
    CheckpointInfo,
    CheckpointManifest,
    CheckpointRank,
)
from shadowspill_client.protocol.codec import encode
from shadowspill_client.protocol.requests import ArtifactRef

from .artifacts import MAX_GROUP_ARTIFACTS, ArtifactStore
from .records.artifact_index import ArtifactAlias


@dataclass(frozen=True)
class RankSave:
    """An exact, quiescent worker description; no exported tensor bytes."""

    rank: int
    size_bytes: int
    state_sha256: str
    program: str
    additional_programs: Mapping[str, str]

    def __post_init__(self) -> None:
        integer(self.size_bytes, "rank archive bytes", minimum=1)
        # Reuse the portable rank identity check, without claiming a payload hash.
        self.reference("preflight", "0" * 64)
        programs = dict(self.additional_programs)
        for value in (self.program, *programs.values()):
            ArtifactRef("program", value, 0)
        if any(not isinstance(key, str) or not key for key in programs):
            raise ValueError("additional checkpoint Programs require names")
        object.__setattr__(self, "additional_programs", MappingProxyType(programs))

    def reference(self, artifact_id: str, sha256: str) -> CheckpointRank:
        return CheckpointRank(
            self.rank,
            ArtifactRef(artifact_id, sha256, self.size_bytes, RANK_MEDIA_TYPE),
            self.state_sha256,
        )

    def confirm(self, info: CheckpointInfo, manifest: Mapping[str, Any]) -> None:
        expected = {
            "format": "shadowspill.checkpoint",
            "version": 2,
            "logical_rank": self.rank,
            "state": self.state_sha256,
            "program": self.program,
            "additional_programs": dict(self.additional_programs),
            "byteorder": info.byteorder,
            "weights": info.weights,
            "frozen_state_id": info.frozen_state_id,
            "metadata": info.rank_metadata(),
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError(
                f"rank {self.rank} checkpoint completion differs from admission"
            )


@dataclass(frozen=True)
class RankSaved:
    """Returned only after a rank's writer and required transfer finish."""

    sha256: str
    manifest: Mapping[str, Any]


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

    return artifacts.produce_group(specs, produce, alias=alias)[-1]
