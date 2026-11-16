"""Private CPU preparation records; never accepted by the public HTTP API."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Literal

from shadowspill.artifacts._values import digest
from shadowspill_client._records import Record, integer, record, text
from shadowspill_client.protocol.checkpoints import (
    CheckpointManifest,
    SaveInfo,
    WeightsManifest,
)
from shadowspill_client.protocol.requests import ArtifactRef

from .checkpoint_records import RankSave, RankSaved


@dataclass(frozen=True)
class ConversionRequest(Record):
    artifact: ArtifactRef
    source: CheckpointManifest | WeightsManifest
    plans: ArtifactRef
    configuration: ArtifactRef
    files: Mapping[str, str]
    directory: str
    host_bytes: int

    def __post_init__(self) -> None:
        for name, kind in (
            ("artifact", ArtifactRef),
            ("plans", ArtifactRef),
            ("configuration", ArtifactRef),
        ):
            object.__setattr__(self, name, record(kind, getattr(self, name), name))
        source = self.source
        weights = isinstance(source, WeightsManifest) or (
            isinstance(source, Mapping)
            and isinstance(source.get("info"), Mapping)
            and source["info"].get("kind") == "weights"
        )
        object.__setattr__(
            self,
            "source",
            record(WeightsManifest, source, "source")
            if weights
            else record(CheckpointManifest, source, "source"),
        )
        if self.artifact.media_type != self.source.info.media_type:
            raise ValueError("conversion source kind differs from its artifact")
        expected = {
            self.source.info.plans.id,
            self.plans.id,
            *(row.archive.id for row in self.source.ranks),
        }
        if not isinstance(self.files, Mapping) or self.files.keys() != expected:
            raise ValueError("conversion requires exactly its granted source files")
        object.__setattr__(
            self,
            "files",
            MappingProxyType(
                {key: text(value, "granted path") for key, value in self.files.items()}
            ),
        )
        text(self.directory, "conversion directory")
        integer(self.host_bytes, "conversion process host envelope", minimum=1)


@dataclass(frozen=True)
class ConversionPrepared(Record):
    info: SaveInfo
    ranks: Sequence[RankSave]
    temporary_bytes: int
    files: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "info", record(SaveInfo, self.info, "checkpoint"))
        ranks = tuple(record(RankSave, row, "rank") for row in self.ranks)
        if tuple(row.rank for row in ranks) != tuple(
            range(self.info.world_size)
        ) or any(row.kind != self.info.kind for row in ranks):
            raise ValueError("conversion requires every target rank in order")
        object.__setattr__(self, "ranks", ranks)
        integer(self.temporary_bytes, "conversion disk bytes")
        integer(self.files, "conversion file count", minimum=1)


@dataclass(frozen=True)
class ConversionCommand(Record):
    action: Literal["write", "close"]

    def __post_init__(self) -> None:
        if self.action not in ("write", "close"):
            raise ValueError("unknown conversion command")


@dataclass(frozen=True)
class ConversionWritten(Record):
    ranks: Sequence[RankSaved]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "ranks",
            tuple(record(RankSaved, row, "saved rank") for row in self.ranks),
        )


@dataclass(frozen=True)
class ConversionFailed(Record):
    message: str

    def __post_init__(self) -> None:
        if len(text(self.message, "conversion failure")) > 2048:
            raise ValueError("conversion failure exceeds its bound")


def weights_conversion_info(request: ConversionRequest, world_size: int) -> SaveInfo:
    """Immutable initialization provenance, without checkpoint resume metadata.

    The source update count remains export provenance. Creating workers from this
    generation starts a fresh run; no optimizer or RNG state is synthesized here.
    """
    if not isinstance(request.source, WeightsManifest):
        raise ValueError("weights initialization requires a weights generation")
    identity = digest(
        {
            "source": request.artifact.to_dict(),
            "plans": request.plans.to_dict(),
            "configuration": request.configuration.to_dict(),
        }
    )
    return replace(
        request.source.info,
        request_id="initialization-" + identity,
        world_size=world_size,
        configuration=request.configuration,
        plans=request.plans,
    )
