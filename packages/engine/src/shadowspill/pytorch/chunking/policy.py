"""Explicit chunk domains for selected effective stages."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from shadowspill.task.inputs import TaskInputRole

from .geometry import ChunkAxis
from .lifetime_policy import ExecutionLifetimes

if TYPE_CHECKING:
    from shadowspill.pytorch.capture.artifacts import TensorGeometry
    from shadowspill.pytorch.partition import PartitionSpec


@dataclass(frozen=True, slots=True)
class AxisFrom:
    """Map an output's chunk axis from an operator input's geometry.

    Remove ``trailing_dims`` feature dimensions before reshaping to the output.
    For example, a normalization's flattened statistics use ``AxisFrom(0, 1)``.
    """

    input_index: int
    trailing_dims: int = 0

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (self.input_index, self.trailing_dims)
        ):
            raise TypeError(
                "axis source positions and dropped dimensions must be integers"
            )
        if self.input_index < 0 or self.trailing_dims < 0:
            raise ValueError(
                "axis source positions and dropped dimensions cannot be negative"
            )


@dataclass(frozen=True, slots=True)
class ChunkLocal:
    """Opaque scratch whose values/coordinates belong to one child invocation.

    ``dim`` identifies a dimension proportional to the logical axis extent;
    ``None`` means fixed geometry. Such values may flow through the child but
    cannot be saved or returned at the parent boundary. An explicit opaque
    rule must restore ordinary axis order before a sliced output can escape.
    """

    dim: int | None = None

    def __post_init__(self) -> None:
        if self.dim is not None and (
            isinstance(self.dim, bool) or not isinstance(self.dim, int)
        ):
            raise TypeError("a chunk-local dimension must be an integer or None")


@dataclass(frozen=True, slots=True)
class OperatorRule:
    """A caller-certified axis contract for one opaque operator overload.

    Each output axis identifies its chunk dimension; ``None`` means shared.
    ``reductions`` identifies outputs that sum contributions across chunks.
    The rule's version participates in executable identity. Rules certify
    semantics, not memory: the complete executable is still measured.
    """

    target: str
    version: str
    output_axes: tuple[int | ChunkAxis | AxisFrom | ChunkLocal | None, ...]
    reductions: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.target or not self.version or not self.output_axes:
            raise ValueError("operator rules require a target, version and outputs")
        if any(
            isinstance(axis, bool)
            or not isinstance(axis, (int, ChunkAxis, AxisFrom, ChunkLocal, type(None)))
            for axis in self.output_axes
        ):
            raise TypeError(
                "operator output axes require a supported axis/local contract"
            )
        if any(isinstance(i, bool) or not isinstance(i, int) for i in self.reductions):
            raise TypeError("operator reduction positions must be integers")
        if any(i < 0 or i >= len(self.output_axes) for i in self.reductions):
            raise ValueError("operator rule reduction index is outside its outputs")
        if any(self.output_axes[i] is not None for i in self.reductions):
            raise ValueError("a reduced output cannot also carry a chunk axis")


@dataclass(frozen=True, slots=True)
class AxisChunks:
    """Chunk a stage's declared input axes, sharing its remaining inputs.

    ``inputs`` maps effective stage input positions to their chunk dimensions.
    All selected inputs describe the same logical extent. Physical dimensions
    may differ and include factors around the selected axis. Unsliced inputs
    are shared, including weights. ``domain`` names the logical coordinates
    so later regions can inherit the mapping. ``variants`` restricts retention
    choices; it never changes ordinary boundary residency planning.
    ``regions`` composes independently sized execution regions inside each
    bounded slice, so their intermediate values never span the complete axis.
    """

    chunk_size: int
    inputs: Mapping[int, int | ChunkAxis]
    variants: tuple[str, ...] = ("save", "recompute")
    rules: tuple[OperatorRule, ...] = ()
    domain: str | None = None
    regions: ChunkRegions | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.chunk_size, int):
            raise TypeError("chunk_size must be an integer")
        if isinstance(self.chunk_size, bool) or self.chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        if not self.inputs or any(
            isinstance(i, bool)
            or not isinstance(i, int)
            or i < 0
            or isinstance(axis, bool)
            or not isinstance(axis, (int, ChunkAxis))
            for i, axis in self.inputs.items()
        ):
            raise ValueError("inputs must map nonnegative positions to integer axes")
        if not self.variants or len(set(self.variants)) != len(self.variants):
            raise ValueError("variants must contain distinct retention choices")
        if set(self.variants) - {"save", "recompute"}:
            raise ValueError("variants must name save or recompute")
        if len({rule.target for rule in self.rules}) != len(self.rules):
            raise ValueError("operator rule targets must be unique")
        if self.domain is not None and (
            not isinstance(self.domain, str) or not self.domain
        ):
            raise ValueError("a chunk domain must be a nonempty string or None")
        if self.regions is not None:
            if not isinstance(self.regions, ChunkRegions):
                raise TypeError("nested regions must be a ChunkRegions policy")
            if self.domain is None:
                raise ValueError("nested regions require a named chunk domain")
            if self.regions.lifetimes is not None:
                raise ValueError(
                    "execution lifetimes belong on outer ChunkRegions, "
                    "not inside AxisChunks.regions"
                )
            if not set(self.variants) <= set(self.regions.variants):
                raise ValueError("nested regions must permit every outer chunk variant")
        object.__setattr__(self, "inputs", MappingProxyType(dict(self.inputs)))


@dataclass(frozen=True, slots=True)
class StageInfo:
    """Read-only structural information for a chunk policy's stage selection."""

    stage_id: str
    module_paths: tuple[str, ...]
    inputs: tuple[TensorGeometry | None, ...]
    input_roles: tuple[TaskInputRole, ...]
    operators: tuple[str, ...]
    terminal: bool
    #: Explicit task direction for an execution region; None for a source stage.
    kind: str | None = None
    checkpoint_stage_id: str | None = None
    structural_contract: str | None = None
    input_domains: tuple[Mapping[str, ChunkAxis], ...] = ()


@runtime_checkable
class ChunkPolicy(Protocol):
    """Return a axis contract for a selected stage, or leave it unchanged."""

    def configure_stage(self, stage: StageInfo) -> AxisChunks | None: ...


@dataclass(frozen=True, slots=True)
class ChunkRegions:
    """Partition explicit tasks inside unchanged checkpoint boundaries.

    ``partition`` receives each explicit forward/backward graph. ``chunking``
    selects chunk domains on its execution regions, including backward regions.
    ``variants`` restricts outer retention choices; a terminal training stage
    keeps its mandatory save form and is left unchanged. Child axis contracts
    must permit every selected outer variant in which they are used.
    ``lifetimes`` optionally moves approved readonly operations on the complete
    task before gradient destinations and execution regions are formed. It is
    accepted only on outer regions and does not change checkpoint retention.
    """

    partition: PartitionSpec
    chunking: ChunkPolicy
    variants: tuple[str, ...] = ("save", "recompute")
    lifetimes: ExecutionLifetimes | None = None

    def __post_init__(self) -> None:
        if self.lifetimes is not None and not isinstance(
            self.lifetimes, ExecutionLifetimes
        ):
            raise TypeError("lifetimes must be ExecutionLifetimes or None")
        if not isinstance(self.chunking, ChunkPolicy):
            raise TypeError("region chunking must implement configure_stage(StageInfo)")
        if (
            not self.variants
            or len(set(self.variants)) != len(self.variants)
            or set(self.variants) - {"save", "recompute"}
        ):
            raise ValueError(
                "region variants must contain distinct save/recompute choices"
            )
