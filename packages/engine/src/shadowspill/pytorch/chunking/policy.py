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
    """An output's chunk axis, derived from one of the operator's inputs.

    The input's axis is carried through the reshape from that input's shape,
    less ``trailing_dims`` trailing dimensions, to the output's shape. A
    normalization's statistics, flattened from ``[batch, rows, features]``,
    use ``AxisFrom(0, 1)``. When the input has no chunk axis, neither has the
    output.

    Parameters
    ----------
    input_index : int
        Position of the input among the operator's positional arguments. At
        least zero.
    trailing_dims : int, default 0
        Trailing dimensions of that input dropped before the reshape. At least
        zero.

    Raises
    ------
    TypeError
        If either is not an int.
    ValueError
        If either is negative.
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
    """Scratch whose values belong to one chunk's invocation only.

    Such values, a dispatch permutation for example, may flow within the
    chunk but are never saved for the backward or returned from the stage.
    An explicit operator rule must restore ordinary axis order before a
    sliced output leaves the chunk.

    Parameters
    ----------
    dim : int, optional
        A dimension whose length is a whole multiple of the chunk's extent.
        None, the default, means the value's shape does not depend on it.

    Raises
    ------
    TypeError
        If ``dim`` is neither an int nor None.
    """

    dim: int | None = None

    def __post_init__(self) -> None:
        if self.dim is not None and (
            isinstance(self.dim, bool) or not isinstance(self.dim, int)
        ):
            raise TypeError("a chunk-local dimension must be an integer or None")


@dataclass(frozen=True, slots=True)
class OperatorRule:
    """An axis contract the caller certifies for one opaque operator overload.

    A chunked stage holding an operator whose behavior along the axis the
    frontend cannot derive fails capture without one. A rule certifies
    semantics, not memory: the chunked task is still measured whole.

    Parameters
    ----------
    target : str
        Exact operator overload name.
    version : str
        Version of the contract. Not empty. It enters executable identity,
        so a changed contract needs a new version.
    output_axes : tuple
        One entry per flattened output. An int names a dimension of exactly
        the axis's extent; a `ChunkAxis` a factored dimension; an `AxisFrom`
        derives the axis from an input; a `ChunkLocal` marks chunk-local
        scratch; None marks an output shared by every chunk, the same in
        each.
    reductions : tuple of int, default ()
        Positions of outputs, each with a None axis, whose values the chunks
        contribute additively: the chunks' values sum to the full result.

    Raises
    ------
    ValueError
        If ``target``, ``version`` or ``output_axes`` is empty, a reduction
        position is outside the outputs, or a reduced output has an axis.
    TypeError
        If an axis entry or a reduction position has an unsupported type.
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
    """Run a stage in slices along one axis of its inputs.

    `ChunkPolicy.configure_stage` returns one for each stage it chunks. The
    stage runs once per slice of at most ``chunk_size`` positions; every
    input ``inputs`` does not name, weights included, is shared by all
    slices.

    Parameters
    ----------
    chunk_size : int
        Most axis positions per invocation. Positive. A shorter last slice
        gets its own compiled form.
    inputs : mapping of int to int or ChunkAxis
        Stage input positions mapped to the dimension that carries the axis;
        an int ``d`` means ``ChunkAxis(d)``. Capture fails unless every named
        input holds the same positive extent of the axis. Stored read-only.
    variants : tuple of str, default ("save", "recompute")
        The stage's graph-pair variants to keep: distinct, from ``"save"``
        and ``"recompute"``. It restricts retention choices, not residency.
    rules : tuple of OperatorRule, default ()
        Contracts for opaque operators in the stage, one per target.
    domain : str, optional
        Name for the axis's coordinates, such as ``"tokens"``. Later regions
        see each value's axis under this name in `StageInfo.input_domains`.
        None keeps the mapping local to this stage.
    regions : ChunkRegions, optional
        Execution regions inside every slice, each with its own chunking, so
        their intermediate values never span the whole axis. Needs a
        ``domain``, must permit every variant in ``variants``, and must not
        set ``lifetimes``.

    Raises
    ------
    TypeError
        If ``chunk_size`` is not an int or ``regions`` is not a
        `ChunkRegions`.
    ValueError
        If ``chunk_size`` is not positive; ``inputs`` is empty or maps a
        negative position or an axis that is neither an int nor a
        `ChunkAxis`; ``variants`` is empty, repeats
        or names another variant; rule targets repeat; ``domain`` is empty;
        or ``regions`` breaks the conditions above.
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
    """The structure of one stage, as a chunk policy sees it.

    Capture passes one to `ChunkPolicy.configure_stage` for each checkpoint
    stage, and under `ChunkRegions` for each execution region.

    Attributes
    ----------
    stage_id : str
        The stage occurrence's identifier. An execution region's is its
        checkpoint stage's followed by ``/region_<index>``.
    module_paths : tuple of str
        Source module paths of the stage's operations, enclosing modules
        included, each once.
    inputs : tuple
        Each input's `TensorGeometry` (shape, strides, dtype and storage
        offset) by position, or None for an input that is not a tensor.
    input_roles : tuple of TaskInputRole
        Each input's role, such as parameter, buffer or activation.
    operators : tuple of str
        Exact operator overload names of the stage's operator calls, in graph
        order.
    terminal : bool
        Whether this is the terminal stage of a training build, which
        chunking must leave whole.
    kind : str or None
        An execution region's task direction, such as ``"forward"`` or
        ``"backward"``; None for a checkpoint stage.
    checkpoint_stage_id : str or None
        An execution region's checkpoint stage; None for a checkpoint stage.
    structural_contract : str or None
        An execution region's structural digest before chunking, for
        overrides keyed by contract; None for a checkpoint stage.
    input_domains : tuple of mapping of str to ChunkAxis
        For each execution-region input, the axes it is known to carry, by
        domain name; empty for a checkpoint stage.
    """

    stage_id: str
    module_paths: tuple[str, ...]
    inputs: tuple[TensorGeometry | None, ...]
    input_roles: tuple[TaskInputRole, ...]
    operators: tuple[str, ...]
    terminal: bool
    kind: str | None = None
    checkpoint_stage_id: str | None = None
    structural_contract: str | None = None
    input_domains: tuple[Mapping[str, ChunkAxis], ...] = ()


@runtime_checkable
class ChunkPolicy(Protocol):
    """What a chunk policy implements: an axis contract for each stage.

    `build_forward`, `build_forward_backward` and `build_step` take one as
    ``chunking`` and call it for each checkpoint stage; `ChunkRegions` calls
    it for each execution region. It must be deterministic and read-only: it
    runs no kernels and changes no capture metadata.
    """

    def configure_stage(self, stage: StageInfo) -> AxisChunks | None:
        """Choose how one stage runs.

        Capture raises `TypeError` for any other return value, and
        `CaptureError` for chunks returned for a terminal stage.

        Parameters
        ----------
        stage : StageInfo
            The stage's structure.

        Returns
        -------
        AxisChunks or None
            The stage's chunking, or None to run it whole.
        """
        ...


@dataclass(frozen=True, slots=True)
class ChunkRegions:
    """Execution regions inside each checkpoint stage, each chunked on its own.

    Passed as ``chunking``, it divides each stage's tasks into regions, with
    checkpoint boundaries and retention left as captured; the terminal stage
    of a training build stays whole. Region intermediates are task workspace:
    they add no checkpoint inputs, saved values or separately scheduled
    tasks.

    Parameters
    ----------
    partition : {"auto", "whole"} or PartitionPolicy
        Divides each task's forward and backward graphs into contiguous
        regions. A `PartitionPolicy` receives the task's graph module as both
        arguments.
    chunking : ChunkPolicy
        Called for each region, backward regions included.
    variants : tuple of str, default ("save", "recompute")
        The outer graph-pair variants to keep: distinct, from ``"save"`` and
        ``"recompute"``. A region's `AxisChunks` must permit each one.
    lifetimes : ExecutionLifetimes, optional
        Operation scheduling on the whole task, applied before gradient
        destinations and regions are formed. Accepted on an outer
        `ChunkRegions` only. None keeps the captured schedule.

    Raises
    ------
    TypeError
        If ``lifetimes`` is not an `ExecutionLifetimes` or ``chunking``
        does not implement `ChunkPolicy`.
    ValueError
        If ``variants`` is empty, repeats or names another variant.
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
