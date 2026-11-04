"""Build-time controls for operation lifetimes within a checkpoint task."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

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
class LifetimeSelection:
    """Optional choices for an operation; unspecified choices remain unset.

    ``move`` permits moving readonly work once toward its first consumer.
    ``barrier`` pins an operation and protects its live internal backing set
    against increases. Barriers take precedence over movement permission.
    Neither choice duplicates computation or requests retained state/residency.
    """

    move: bool | None = None
    barrier: bool | None = None

    def __post_init__(self) -> None:
        if any(
            value is not None and type(value) is not bool
            for value in (self.move, self.barrier)
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
    source_target: str | None = field(default=None, kw_only=True)
    module_pattern: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        _name(self.target, "lifetime rule target")
        _name(self.source_target, "source target", optional=True)
        _name(self.module_pattern, "module pattern", optional=True)
        LifetimeSelection(self.move, self.barrier)
        if self.move is None and self.barrier is None:
            raise ValueError("a lifetime rule must set move or barrier")


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


@dataclass(frozen=True, slots=True)
class ExecutionLifetimes:
    """Control movement once within an outer ``ChunkRegions`` task.

    Resolve rules and ``select`` at build time, before gradient destinations
    and execution regions are formed. The callback runs once per operation in
    that occurrence and returns immutable choices or ``None``. Only resolved
    choices, effects contracts and scheduling decisions enter the executable
    identity; callbacks and source module names never do. Unknown effects are
    fences, and protected backing sets may not increase. Compilation/profiling
    measures the final composition; ordinary planning owns physical admission.

    This policy does not enable local replay or storage compaction. Expensive
    operations may move when certified, but are never duplicated by it.
    """

    rules: tuple[LifetimeRule, ...] = ()
    readonly: tuple[ReadonlyOperator, ...] = ()
    select: Callable[[LifetimeOperation], LifetimeSelection | None] | None = None

    def __post_init__(self) -> None:
        for values, expected, label in (
            (self.rules, LifetimeRule, "rules"),
            (self.readonly, ReadonlyOperator, "readonly"),
        ):
            if not isinstance(values, tuple) or any(
                not isinstance(value, expected) for value in values
            ):
                raise TypeError(
                    f"{label} must be a tuple of {expected.__name__} values"
                )
        if len({item.target for item in self.readonly}) != len(self.readonly):
            raise ValueError("readonly operator targets must be unique")
        if self.select is not None and not callable(self.select):
            raise TypeError("lifetime select must be callable or None")
