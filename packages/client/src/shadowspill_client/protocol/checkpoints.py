"""Portable checkpoint provenance and the complete immutable rank generation."""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from .._records import Record, Value, integer, mapping, record, text
from .frozen import FrozenBase
from .requests import ArtifactRef, identifier

CHECKPOINT_MEDIA_TYPE = "application/vnd.shadowspill.checkpoint+json"
RANK_MEDIA_TYPE = "application/vnd.shadowspill.checkpoint-rank"


@dataclass(frozen=True)
class CheckpointConversion(Record):
    """Immutable source and transformation for a materialized target generation."""

    source: ArtifactRef
    source_plans: ArtifactRef
    logical_state_sha256: str
    version: Literal[1] = 1
    randomness: Mapping[str, Value] | None = None

    def __post_init__(self) -> None:
        for name in ("source", "source_plans"):
            object.__setattr__(
                self, name, record(ArtifactRef, getattr(self, name), name)
            )
        if self.source.media_type != CHECKPOINT_MEDIA_TYPE:
            raise ValueError("conversion source requires a checkpoint generation")
        if type(self.version) is not int or self.version != 1:
            raise ValueError("unsupported checkpoint conversion version")
        if (
            not isinstance(self.logical_state_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.logical_state_sha256) is None
        ):
            raise ValueError("conversion requires a logical state SHA-256 identity")
        if self.randomness is not None:
            object.__setattr__(
                self, "randomness", mapping(self.randomness, "RNG conversion")
            )


@dataclass(frozen=True)
class CheckpointInfo(Record):
    """The completed boundary and fixed setup shared by every saved rank.

    Caller metadata is explicit JSON. Configuration retains setup defaults;
    per-call optimizer or objective overrides never replace those defaults.
    Frozen state is an exact external initialization identity, not a local path.
    """

    session_id: str
    run_id: str
    history_id: str
    request_id: str
    completed_updates: int
    world_size: int
    configuration: ArtifactRef
    plans: ArtifactRef
    byteorder: Literal["little", "big"]
    weights: Literal["master", "compute"] = "master"
    frozen_state_id: str | None = None
    metadata: Mapping[str, Value] = field(default_factory=dict)
    conversion: CheckpointConversion | None = None
    frozen_base: FrozenBase | None = None

    def __post_init__(self) -> None:
        for name in ("session_id", "run_id", "history_id", "request_id"):
            identifier(getattr(self, name), name)
        integer(self.completed_updates, "completed updates")
        integer(self.world_size, "checkpoint worker count", minimum=1)
        if self.byteorder not in ("little", "big"):
            raise ValueError("unknown checkpoint byte order")
        if self.weights not in ("master", "compute"):
            raise ValueError("unknown checkpoint weight representation")
        for name in ("configuration", "plans"):
            object.__setattr__(
                self, name, record(ArtifactRef, getattr(self, name), name)
            )
        if self.frozen_state_id is not None:
            text(self.frozen_state_id, "frozen state identity")
        object.__setattr__(
            self, "metadata", mapping(self.metadata, "checkpoint metadata")
        )
        if self.conversion is not None:
            object.__setattr__(
                self,
                "conversion",
                record(CheckpointConversion, self.conversion, "conversion"),
            )
        if self.frozen_base is not None:
            object.__setattr__(
                self, "frozen_base", record(FrozenBase, self.frozen_base, "frozen base")
            )
            if self.frozen_state_id != self.frozen_base.identity:
                raise ValueError("frozen state identity differs from its exact base")

    def rank_metadata(self) -> dict[str, object]:
        """The same provenance travels inside each ordinary rank archive."""
        value = self.to_dict()
        if self.conversion is None:
            # Ordinary generations retain the existing rank metadata format.
            value.pop("conversion")
        if self.frozen_base is None:
            value.pop("frozen_base")
        return {"checkpoint": value}


@dataclass(frozen=True)
class CheckpointRank(Record):
    rank: int
    archive: ArtifactRef
    state_sha256: str

    def __post_init__(self) -> None:
        integer(self.rank, "checkpoint rank")
        object.__setattr__(
            self, "archive", record(ArtifactRef, self.archive, "archive")
        )
        if self.archive.media_type != RANK_MEDIA_TYPE:
            raise ValueError("checkpoint rank requires a rank archive")
        if (
            not isinstance(self.state_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.state_sha256) is None
        ):
            raise ValueError("checkpoint state requires a SHA-256 identity")


@dataclass(frozen=True)
class CheckpointManifest(Record):
    info: CheckpointInfo
    ranks: Sequence[CheckpointRank]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "info", record(CheckpointInfo, self.info, "checkpoint")
        )
        if not isinstance(self.ranks, (tuple, list)) or not self.ranks:
            raise ValueError("checkpoint requires its complete rank set")
        ranks = tuple(record(CheckpointRank, rank, "rank") for rank in self.ranks)
        if len(ranks) != self.info.world_size or any(
            rank.rank != index for index, rank in enumerate(ranks)
        ):
            raise ValueError(
                "checkpoint requires each logical rank exactly once in order"
            )
        if len({rank.archive.id for rank in ranks}) != len(ranks):
            raise ValueError("checkpoint ranks must reference distinct archives")
        object.__setattr__(self, "ranks", ranks)
