"""Build-time controls for operation lifetimes within a checkpoint task."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from shadowspill.pytorch.capture.artifacts import TensorGeometry


def _name(value: str | None, label: str, *, optional: bool = False) -> None:
    if value is None and optional:
        return
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")


@dataclass(frozen=True, slots=True)
class ReadonlyOperator:
    """Versioned assertion of deterministic readonly opaque execution.

    This cannot override dispatcher writes, RNG/effect registrations, or an
    unknown alias-return contract. It certifies effects, not axis independence,
    computational cost or physical memory use.
    """

    target: str
    version: str

    def __post_init__(self) -> None:
        _name(self.target, "readonly operator target")
        _name(self.version, "readonly operator version")


@dataclass(frozen=True, slots=True)
class LayoutIndependentOperator:
    """Versioned value semantics for relayout of opaque input arguments.

    ``inputs`` contains dispatcher argument indices, including keyword-only
    arguments and entire tensor lists. Replacing those inputs by strided tensors
    with equal logical shape/dtype/device/values must preserve all logical output
    values, shapes and dtypes, even when their strides, offsets, storage extents
    or alias relationships differ. Every logical output element is initialized.
    Output layout may change and is checked transitively. This does not certify
    effects, numerical bitwise invariance, axis independence or memory use.

    A separate readonly contract is required. Mutation/RNG and unqualified ATen
    overloads cannot be overridden. These contracts are resolved during build,
    never during execution.
    """

    target: str
    version: str
    inputs: tuple[int, ...]

    def __post_init__(self) -> None:
        _name(self.target, "layout operator target")
        _name(self.version, "layout operator version")
        if (
            not isinstance(self.inputs, tuple)
            or not self.inputs
            or any(type(index) is not int or index < 0 for index in self.inputs)
            or len(set(self.inputs)) != len(self.inputs)
        ):
            raise ValueError(
                "layout inputs must be unique nonnegative argument indices"
            )


@dataclass(frozen=True, slots=True)
class LifetimeSelection:
    """Optional choices for an operation; unspecified choices remain unset.

    ``move`` permits moving readonly work once toward its first consumer.
    ``barrier`` pins an operation and protects its live internal storage set
    against increases. Barriers take precedence over movement permission.
    ``compact`` permits preparing and copying an internal view for consumers
    in later execution regions, subject to layout and complete storage proofs.
    ``replay`` permits repeating this primal computation in consumer regions.
    No arithmetic is duplicated without explicit replay permission. These
    choices do not request checkpoint retention or residency.
    """

    move: bool | None = None
    barrier: bool | None = None
    compact: bool | None = None
    replay: bool | None = None

    def __post_init__(self) -> None:
        if any(
            value is not None and type(value) is not bool
            for value in (self.move, self.barrier, self.compact, self.replay)
        ):
            raise TypeError("lifetime choices must be bool or None")


@dataclass(frozen=True, slots=True)
class LifetimeRule:
    """Select an exact explicit operator overload, optionally by source scope.

    ``source_target`` matches the original operator from which capture emitted
    the operation. ``module_pattern`` matches a case-sensitive shell pattern
    against any source module path. Both require actual source attribution.
    Matching rules and callbacks must agree on each explicitly set choice.
    """

    target: str
    move: bool | None = field(default=None, kw_only=True)
    barrier: bool | None = field(default=None, kw_only=True)
    compact: bool | None = field(default=None, kw_only=True)
    replay: bool | None = field(default=None, kw_only=True)
    source_target: str | None = field(default=None, kw_only=True)
    module_pattern: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        _name(self.target, "lifetime rule target")
        _name(self.source_target, "source target", optional=True)
        _name(self.module_pattern, "module pattern", optional=True)
        selection = LifetimeSelection(
            self.move, self.barrier, self.compact, self.replay
        )
        if all(
            value is None
            for value in (
                selection.move,
                selection.barrier,
                selection.compact,
                selection.replay,
            )
        ):
            raise ValueError(
                "a lifetime rule must set move, barrier, compact or replay"
            )


@dataclass(frozen=True, slots=True)
class LifetimeOperation:
    """Immutable occurrence information passed to a build-time selector.

    Geometry describes flattened input/output leaves without live tensors or
    mutable FX nodes. Indices refer to the complete explicit task before any
    lifetime movement or execution partitioning.
    """

    index: int
    target: str
    source_target: str | None
    module_paths: tuple[str, ...]
    effect_fence: str | None
    kind: str
    variant: str | None
    stage_id: str
    checkpoint_stage_id: str
    inputs: tuple[TensorGeometry | None, ...]
    outputs: tuple[TensorGeometry | None, ...]
    phase: Literal["primal", "derivative", "unknown"] = "unknown"


@dataclass(frozen=True, slots=True)
class ExecutionLifetimes:
    """Control operation lifetimes in an outer ``ChunkRegions`` task.

    Resolve rules and ``select`` at build time, before gradient destinations
    and execution regions are formed. The callback runs once per operation in
    that occurrence and returns immutable choices or ``None``. Only resolved
    choices, effects contracts and scheduling decisions enter the executable
    identity; callbacks and source module names never do. Unknown effects are
    fences, and protected storage sets may not increase. Compilation/profiling
    measures the final composition; ordinary planning owns physical admission.

    Compaction prepares selected internal aliases at their producer and considers
    escaping sibling views together. It needs a strict storage reduction at one
    selected barrier and no increase at any other. Layout-sensitive consumers
    require separately qualified input-value semantics. Unsupported candidates
    remain un-copied with recorded reasons. With no barriers, no copy is useful
    under this policy. Input/external aliases and written backings never compact.

    Replay has no default selections or implicit cost-based permissions. An
    explicit ``replay=True`` allows duplication only for certified immutable
    primal dependencies. Structural tuple extraction accompanies a selected
    producer unless explicitly excluded. Copies are shared within each consumer
    region; regions containing barriers never receive them. Changed dependency
    versions, externally observed/written outputs and gradient conversions are
    protected. Value/layout contracts also guard changed alias relationships.
    Expensive work is duplicated only if explicitly selected, and the final
    measured composition includes its cost. ``move=False`` does not pin a view
    selected for compaction preparation; ``barrier=True`` pins an operation.
    """

    rules: tuple[LifetimeRule, ...] = ()
    readonly: tuple[ReadonlyOperator, ...] = ()
    select: Callable[[LifetimeOperation], LifetimeSelection | None] | None = None
    layout: tuple[LayoutIndependentOperator, ...] = field(default=(), kw_only=True)

    def __post_init__(self) -> None:
        for values, expected, label in (
            (self.rules, LifetimeRule, "rules"),
            (self.readonly, ReadonlyOperator, "readonly"),
            (self.layout, LayoutIndependentOperator, "layout"),
        ):
            if not isinstance(values, tuple) or any(
                not isinstance(value, expected) for value in values
            ):
                raise TypeError(
                    f"{label} must be a tuple of {expected.__name__} values"
                )
        if len({item.target for item in self.readonly}) != len(self.readonly):
            raise ValueError("readonly operator targets must be unique")
        if len({item.target for item in self.layout}) != len(self.layout):
            raise ValueError("layout operator targets must be unique")
        if self.select is not None and not callable(self.select):
            raise TypeError("lifetime select must be callable or None")
