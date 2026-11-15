"""Checkpoint descriptions shared by private workers and their coordinator."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from shadowspill_client._records import integer
from shadowspill_client.protocol.checkpoints import (
    RANK_MEDIA_TYPE,
    CheckpointInfo,
    CheckpointRank,
)
from shadowspill_client.protocol.requests import ArtifactRef


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


