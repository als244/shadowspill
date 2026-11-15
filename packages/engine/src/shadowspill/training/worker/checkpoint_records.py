"""Checkpoint descriptions shared by private workers and their coordinator."""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from shadowspill_client._records import Record, Value, integer, mapping, record
from shadowspill_client.protocol.checkpoints import (
    CHECKPOINT_MEDIA_TYPE,
    RANK_MEDIA_TYPE,
    CheckpointInfo,
    CheckpointRank,
)
from shadowspill_client.protocol.requests import ArtifactRef


@dataclass(frozen=True)
class RankRestore(Record):
    checkpoint: ArtifactRef
    info: CheckpointInfo
    rank: CheckpointRank

    def __post_init__(self) -> None:
        for name, kind in (
            ("checkpoint", ArtifactRef),
            ("info", CheckpointInfo),
            ("rank", CheckpointRank),
        ):
            object.__setattr__(self, name, record(kind, getattr(self, name), name))
        if (
            self.checkpoint.media_type != CHECKPOINT_MEDIA_TYPE
            or self.rank.rank >= self.info.world_size
        ):
            raise ValueError("restore rank differs from its checkpoint generation")


@dataclass(frozen=True)
class RankSave(Record):
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
        expected = mapping(self.confirmation(info), "checkpoint confirmation")
        actual = mapping(
            {key: manifest.get(key) for key in expected}, "rank completion"
        )
        if actual != expected:
            raise ValueError(
                f"rank {self.rank} checkpoint completion differs from admission"
            )

    def confirmation(self, info: CheckpointInfo) -> dict[str, Any]:
        """Bounded completion fields; the full tensor manifest stays in the archive."""
        return {
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


@dataclass(frozen=True)
class RankSaved(Record):
    """Returned only after a rank's writer and required transfer finish."""

    sha256: str
    manifest: Mapping[str, Value]

    def __post_init__(self) -> None:
        ArtifactRef("completion", self.sha256, 0)
        object.__setattr__(self, "manifest", mapping(self.manifest, "rank completion"))


def checkpoint_id(info: CheckpointInfo) -> str:
    key = "\0".join((info.session_id, info.run_id, info.request_id))
    return "checkpoint." + hashlib.sha256(key.encode()).hexdigest()
