"""Build-time controls for operation lifetimes within a checkpoint task."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from shadowspill.pytorch.capture.artifacts import TensorGeometry


def _validate_name(value: str | None, label: str, *, optional: bool = False) -> None:
    if value is None and optional:
        return
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")


@dataclass(frozen=True, slots=True)
class ReadonlyOperator:
    """An effect contract: an operator overload is deterministic and only reads.

    `ExecutionLifetimes` treats an operator outside the ``aten`` namespace as
    an effect fence unless its ``readonly`` holds one of these: no operation
    moves or is replayed across a fence, and a fence is itself neither moved,
    replayed nor compacted. A contract for an operator whose schema writes an
    input, or that is registered as random or effectful, fails capture with
    `CaptureError`. It certifies effects only, not axis independence, cost or
    memory.

    Parameters
    ----------
    target : str
        Exact operator overload name, such as ``"provider.attention.default"``.
        Not blank.
    version : str
        Version of the contract. Not blank. It enters executable identity, so
        a changed contract needs a new version.

    Raises
    ------
    ValueError
        If ``target`` or ``version`` is not a string or is blank.
    """

    target: str
    version: str

    def __post_init__(self) -> None:
        _validate_name(self.target, "readonly operator target")
        _validate_name(self.version, "readonly operator version")


@dataclass(frozen=True, slots=True)
class LayoutIndependentOperator:
    """A layout contract: inputs an operator overload reads by value only.

    Compaction and replay can hand an operator an input copied into another
    storage layout. The contract certifies that, for each listed input, any
    strided tensor of equal shape, dtype, device and values, whatever its
    strides, offset, storage extent or aliasing, gives the same output values,
    shapes and dtypes, with every output element written; bitwise equality is
    not promised. The outputs' own layouts may change, and capture follows the
    change through them. It applies to operators outside the ``aten``
    namespace, and needs a `ReadonlyOperator` for the same target: a change
    whose layout reaches an effect fence is refused.

    Parameters
    ----------
    target : str
        Exact operator overload name, outside the ``aten`` namespace. Not
        blank.
    version : str
        Version of the contract. Not blank. Where the contract is used, it
        enters executable identity, so a changed contract needs a new version.
    inputs : tuple of int
        Positions in the operator's schema of the inputs it covers, keyword-only
        arguments included; a list of tensors is one input. Not empty,
        distinct, each at least zero.

    Raises
    ------
    ValueError
        If ``target`` or ``version`` is not a string or is blank, ``target``
        names an ``aten`` operator, or ``inputs`` is not a tuple meeting the
        conditions above.
    """

    target: str
    version: str
    inputs: tuple[int, ...]

    def __post_init__(self) -> None:
        _validate_name(self.target, "layout operator target")
        _validate_name(self.version, "layout operator version")
        if self.target.split(".", 1)[0] == "aten":
            # Capture knows which ATen inputs are read by value; it reads no
            # contract for them.
            raise ValueError(
                "layout contracts name operators outside the aten namespace, "
                f"not {self.target!r}"
            )
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
    """Lifetime choices for one operation.

    An `ExecutionLifetimes` ``select`` callable returns one, and a
    `LifetimeRule` carries the same choices. None leaves a choice to the other
    rules; two that set one choice differently fail capture with
    `CaptureError`. A permitted change is still left undone where it would
    grow a barrier's live storage, as `ExecutionLifetimes` describes. No
    choice saves a value or requests a residency.

    Parameters
    ----------
    move : bool, optional
        True lets the operation move once, to just before its first consumer,
        when it crosses no effect fence other than an ATen write to storage it
        does not use. Barriers stay in place, as do the dtype conversions from
        which a forward task's outputs or a backward task's parameter
        gradients are assembled. Selecting an effect fence fails capture with
        `CaptureError`.
    barrier : bool, optional
        True makes the operation a barrier: it stays in place and is neither
        compacted nor replayed, no change may grow its live storage, and no
        replayed copy enters its execution region.
    compact : bool, optional
        True, for a view of the task's own storage, moves the view to just
        after its last input and copies it into contiguous storage for its
        consumers in other execution regions; views of one storage are copied
        together. The copy is kept only when it shrinks some barrier's live
        storage (so without barriers nothing is compacted) and the changed
        layout reaches no task output or effect fence, only operations that
        read it by value: known ATen operators and those a
        `LayoutIndependentOperator` covers. Views of an input's or a task
        output's storage are left alone, and views of written storage or as
        large as their storage are not copied.
    replay : bool, optional
        True lets each other execution region that consumes this forward
        (``"primal"``) operation compute its own copy, once per region,
        instead of reading the original result. Only operations with
        ``replay`` True are copied, with the tuple extraction of their
        results; False excludes an operation, that extraction included.
        Replay needs an operation that is not an effect fence, whose result's
        storage is neither a task output's nor written, and whose inputs still
        hold their values where the copy runs; any layout it changes must meet
        the condition of ``compact``.

    Raises
    ------
    TypeError
        If a choice is neither a bool nor None.
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
    """Lifetime choices for the operations of one operator overload.

    `ExecutionLifetimes` takes rules as ``rules``. Every rule that matches an
    operation applies; two that set one choice differently, or a rule and the
    ``select`` callable, fail capture with `CaptureError`.

    Parameters
    ----------
    target : str
        Exact operator overload name in the task, such as
        ``"aten.slice.Tensor"``. Not blank.
    move, barrier, compact, replay : bool, optional
        The choices, as in `LifetimeSelection`. At least one is set. Keyword
        only.
    source_target : str, optional
        The operator of the exported model that the operation derives from;
        the rule matches only where it is this one. Not blank. None, the
        default, matches any. Keyword only.
    module_pattern : str, optional
        Case-sensitive shell pattern, such as ``"model.blocks.*"``, that one of
        the operation's source module paths must match. Not blank. None, the
        default, matches any. Keyword only.

    Raises
    ------
    ValueError
        If ``target``, or a ``source_target`` or ``module_pattern`` that is
        not None, is not a string or is blank, or no choice is set.
    TypeError
        If a choice is neither a bool nor None.
    """

    target: str
    move: bool | None = field(default=None, kw_only=True)
    barrier: bool | None = field(default=None, kw_only=True)
    compact: bool | None = field(default=None, kw_only=True)
    replay: bool | None = field(default=None, kw_only=True)
    source_target: str | None = field(default=None, kw_only=True)
    module_pattern: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        _validate_name(self.target, "lifetime rule target")
        _validate_name(self.source_target, "source target", optional=True)
        _validate_name(self.module_pattern, "module pattern", optional=True)
        selection = LifetimeSelection(
            move=self.move,
            barrier=self.barrier,
            compact=self.compact,
            replay=self.replay,
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
    """One operation of a task, as an `ExecutionLifetimes` ``select`` sees it.

    Capture passes one to the callable for each operation of each task, before
    any lifetime change. It holds no tensors or graph nodes.

    Attributes
    ----------
    index : int
        The operation's position among all nodes of the task graph, its inputs
        included.
    target : str
        Exact operator overload name.
    source_target : str or None
        The operator of the exported model that the operation derives from;
        None when capture recorded none.
    module_paths : tuple of str
        Source module paths of the operation, enclosing modules included, each
        once.
    effect_fence : str or None
        Why the operation is an effect fence: ``"ordered mutation"``,
        ``"RNG or other observable effect"`` or
        ``"opaque operator requires a readonly effect contract"``. None when
        it is not one.
    kind : str
        The task's direction: ``"forward"``, ``"backward"`` or
        ``"inference"``.
    variant : str or None
        The graph-pair variant, ``"save"`` or ``"recompute"``; None in a
        forward build.
    stage_id : str
        The checkpoint stage's identifier.
    checkpoint_stage_id : str
        The checkpoint stage's identifier, the same as ``stage_id``.
    inputs : tuple
        Each flattened argument's `TensorGeometry`, or None for an argument
        that is not a tensor.
    outputs : tuple
        Each flattened result's `TensorGeometry`, or None for a result that is
        not a tensor.
    phase : {"primal", "derivative", "unknown"}
        ``"primal"`` for forward computation, recomputation included;
        ``"derivative"`` for backward computation; ``"unknown"`` when capture
        recorded neither.
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
    """Lifetime changes within each task of an outer `ChunkRegions`.

    Passed as `ChunkRegions.lifetimes`, it can move operations toward their
    consumers, copy views into smaller storage for later execution regions,
    and repeat forward work in the regions that use it, each only where the
    rules or ``select`` permit, as `LifetimeSelection` describes. It acts on
    each task `ChunkRegions` divides, before gradient destinations and
    execution regions are formed: moves first, then replay, then compaction.
    It changes neither the saved-state cut nor where values reside, and
    profiling measures the result.

    No change may grow a barrier's live storage: the bytes of the task's own
    intermediate storage produced before the barrier and read at or after
    it, not counting inputs, operator workspace or allocator overhead. A
    change that fails a check is left undone, with its reason recorded in the
    task's metadata. The resolved choices, the contract versions used and the
    decisions taken enter executable identity; the ``select`` callable and
    module paths do not. Capture fails with `CaptureError` when a task holds
    anything but dispatcher operator calls and tuple extraction.

    Parameters
    ----------
    rules : tuple of LifetimeRule, default ()
        Choices by operator overload.
    readonly : tuple of ReadonlyOperator, default ()
        Effect contracts, at most one per target.
    select : callable, optional
        Called once for each operation of each task with its
        `LifetimeOperation`, during build only; returns a `LifetimeSelection`,
        whose choices combine with the rules', or None. Capture fails with
        `CaptureError` for anything else. It must be deterministic. None, the
        default, adds no choices.
    layout : tuple of LayoutIndependentOperator, default ()
        Layout contracts, at most one per target, each with a ``readonly``
        contract for the same target. Keyword only.

    Raises
    ------
    TypeError
        If ``rules``, ``readonly`` or ``layout`` is not a tuple of its type, or
        ``select`` is neither callable nor None.
    ValueError
        If two ``readonly`` or two ``layout`` contracts share a target, or a
        ``layout`` contract's target has no ``readonly`` contract.
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
        # Without an effect contract the operator is a fence, which no changed
        # layout may reach, so its layout contract would never be read.
        uncertified = {item.target for item in self.layout} - {
            item.target for item in self.readonly
        }
        if uncertified:
            raise ValueError(
                "layout contracts need a readonly contract for the same target: "
                f"{sorted(uncertified)}"
            )
        if self.select is not None and not callable(self.select):
            raise TypeError("lifetime select must be callable or None")
