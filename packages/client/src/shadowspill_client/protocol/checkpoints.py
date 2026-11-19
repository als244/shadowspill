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
    """Where a converted checkpoint came from.

    A run started from a checkpoint saved under other Plans loads a copy that
    the service converts for the run's Plans and worker count; the copy's
    `SaveInfo.conversion` records its source.

    Parameters
    ----------
    source : ArtifactRef
        The manifest of the checkpoint the copy was converted from.
    source_plans : ArtifactRef
        The Plans the source was saved under.
    logical_state_sha256 : str
        SHA-256 of the training state's logical description, which the source
        and the copy share: each value's role, full shape, data type and ties,
        whatever its division among workers.
    version : int, default 1
        Version of this record's format.
    randomness : mapping of str to object, optional
        How the source's random streams map onto the copy's workers: the root
        seed, and each source and target stream with whether it is preserved
        or derived. None when the source saved no random state.
    """

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
    """The provenance and setup of a saved checkpoint or weights export.

    Every worker's file holds the same record.

    Parameters
    ----------
    session_id : str
        The session of the run that saved it.
    run_id : str
        The run that saved it.
    history_id : str
        The training history at the save.
    request_id : str
        ID of the save request. A copy the service converted for a run under
        other Plans has an ID the service derived instead.
    completed_updates : int
        Optimizer updates completed at the save.
    world_size : int
        Number of workers, each with its own file.
    configuration : ArtifactRef
        The run's resolved training configuration, with its setup defaults;
        per-call optimizer and objective controls do not replace them.
    plans : ArtifactRef
        The Plans the state was saved under.
    byteorder : str
        ``"little"`` or ``"big"``: the saving host's byte order. A run on a
        host of the other order refuses the state.
    weights : str, default "master"
        ``"master"`` (the master weights) or ``"compute"`` (the weights at
        compute precision), as chosen at the save.
    frozen_state_id : str, optional
        Identity of ``frozen_base``, when the files leave out frozen
        parameters that no Plan changes; None otherwise.
    metadata : mapping of str to object, default {}
        The caller metadata given at the save.
    conversion : CheckpointConversion, optional
        For a checkpoint the service converted for a run under other Plans,
        where it came from; None otherwise.
    frozen_base : FrozenBase, optional
        How a run re-creates the frozen parameters the files leave out: the
        model settings with their initialization, the model definition's
        source and, for weights initialized from an artifact, that artifact.
        None when the files leave none out.
    kind : str, default "checkpoint"
        ``"checkpoint"``, or ``"weights"`` for a weights export.
    format : str, default "shadowspill"
        The file format, always ``"shadowspill"``.
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
        """Return the provenance each worker's file holds.

        Returns
        -------
        dict
            This record's fields under its ``kind`` as the only key, without
            ``conversion`` or ``frozen_base`` when they are None, and without
            ``kind`` and ``format`` for a checkpoint.
        """
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
        """The manifest's media type, which differs by ``kind``."""
        return (
            CHECKPOINT_MEDIA_TYPE if self.kind == "checkpoint" else WEIGHTS_MEDIA_TYPE
        )

    @property
    def rank_media_type(self) -> str:
        """The media type of each worker's file, which differs by ``kind``."""
        return RANK_MEDIA_TYPE if self.kind == "checkpoint" else WEIGHTS_RANK_MEDIA_TYPE


@dataclass(frozen=True)
class SavedRank(Record):
    """One worker's file in a saved checkpoint or weights export.

    Parameters
    ----------
    rank : int
        The worker's rank, from zero.
    archive : ArtifactRef
        The file.
    state_sha256 : str
        SHA-256 of the worker's state inventory: the name, shape, data type
        and layout of each persistent tensor the worker holds, and the part
        it owns. It identifies which state the file holds, not its values,
        and is the same under any Plans that train that state.
    """

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
    """The manifest of a saved checkpoint: its provenance and one file per worker.

    `CheckpointHandle.manifest` returns it, and `GenerationHandle.download`
    writes it as ``manifest.json``.

    Parameters
    ----------
    info : SaveInfo
        The save's provenance and setup.
    ranks : sequence of SavedRank
        One entry per worker, in rank order.
    """

    info: SaveInfo
    ranks: Sequence[SavedRank]

    def __post_init__(self) -> None:
        info, ranks = _generation(self.info, self.ranks, "checkpoint")
        object.__setattr__(self, "info", info)
        object.__setattr__(self, "ranks", ranks)


@dataclass(frozen=True)
class WeightsManifest(Record):
    """The manifest of exported weights: their provenance and one file per worker.

    `WeightsHandle.manifest` returns it, and `GenerationHandle.download`
    writes it as ``manifest.json``. The files hold model weights and
    persistent buffers, never optimizer state. When they leave out frozen
    parameters, ``info.frozen_base`` is set.

    Parameters
    ----------
    info : SaveInfo
        The export's provenance and setup.
    ranks : sequence of SavedRank
        One entry per worker, in rank order.
    """

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
