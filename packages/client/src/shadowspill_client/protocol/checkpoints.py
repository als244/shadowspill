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
WEIGHTS_MEDIA_TYPE = "application/vnd.shadowspill.weights+json"
WEIGHTS_RANK_MEDIA_TYPE = "application/vnd.shadowspill.weights-rank"


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
class SaveInfo(Record):
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
    kind: Literal["checkpoint", "weights"] = "checkpoint"
    format: str = "shadowspill"

    def __post_init__(self) -> None:
        if self.kind not in ("checkpoint", "weights") or self.format != "shadowspill":
            raise ValueError("unknown saved-state kind or archive format")
        if self.kind == "weights" and self.conversion is not None:
            raise ValueError("weights exports have no optimizer-state conversion")
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
        if self.kind == "checkpoint":
            value.pop("kind")
            value.pop("format")
        return {self.kind: value}

    @property
    def media_type(self) -> str:
        return (
            CHECKPOINT_MEDIA_TYPE if self.kind == "checkpoint" else WEIGHTS_MEDIA_TYPE
        )

    @property
    def rank_media_type(self) -> str:
        return RANK_MEDIA_TYPE if self.kind == "checkpoint" else WEIGHTS_RANK_MEDIA_TYPE


@dataclass(frozen=True)
class SavedRank(Record):
    rank: int
    archive: ArtifactRef
    state_sha256: str

    def __post_init__(self) -> None:
        integer(self.rank, "checkpoint rank")
        object.__setattr__(
            self, "archive", record(ArtifactRef, self.archive, "archive")
        )
        if self.archive.media_type not in (RANK_MEDIA_TYPE, WEIGHTS_RANK_MEDIA_TYPE):
            raise ValueError("checkpoint rank requires a rank archive")
        if (
            not isinstance(self.state_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.state_sha256) is None
        ):
            raise ValueError("checkpoint state requires a SHA-256 identity")


@dataclass(frozen=True)
class CheckpointManifest(Record):
    info: SaveInfo
    ranks: Sequence[SavedRank]

    def __post_init__(self) -> None:
        info, ranks = _generation(self.info, self.ranks, "checkpoint")
        object.__setattr__(self, "info", info)
        object.__setattr__(self, "ranks", ranks)


@dataclass(frozen=True)
class WeightsManifest(Record):
    """Inference model state and base dependencies; never optimizer-resume state."""

    info: SaveInfo
    ranks: Sequence[SavedRank]

    def __post_init__(self) -> None:
        info, ranks = _generation(self.info, self.ranks, "weights")
        if info.frozen_state_id is not None and info.frozen_base is None:
            raise ValueError("weights export requires its exact frozen base")
        object.__setattr__(self, "info", info)
        object.__setattr__(self, "ranks", ranks)


def _generation(
    info: object, rows: object, kind: str
) -> tuple[SaveInfo, tuple[SavedRank, ...]]:
    info = record(SaveInfo, info, kind)
    if info.kind != kind:
        raise ValueError("saved state kind differs from its generation manifest")
    if not isinstance(rows, (tuple, list)) or not rows:
        raise ValueError(f"{kind} requires its complete rank set")
    ranks = tuple(record(SavedRank, row, "rank") for row in rows)
    if len(ranks) != info.world_size or any(
        row.rank != index for index, row in enumerate(ranks)
    ):
        raise ValueError(f"{kind} requires each logical rank exactly once in order")
    if any(row.archive.media_type != info.rank_media_type for row in ranks):
        raise ValueError("rank archive kind differs from its generation")
    if len({row.archive.id for row in ranks}) != len(ranks):
        raise ValueError(f"{kind} ranks must reference distinct archives")
    return info, ranks
