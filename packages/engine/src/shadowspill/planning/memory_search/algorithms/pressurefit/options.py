"""What PressureFit is told: its candidate space and how hard it tries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from fractions import Fraction
from typing import ClassVar

from shadowspill.planning.memory_search.request import OptionRecord
from shadowspill.planning.memory_search.toolkit.resolution import (
    DEFAULT_RESOLUTION_OPTIONS,
    validate_resolution_options,
    validate_tail_save_counts,
)

#: The residency strategies PressureFit knows how to build a candidate for.
RESIDENCY_STRATEGIES = (
    "headroom-stall",
    "headroom-transfer",
    "tight-stall",
    "tight-transfer",
)
#: The fetch-trigger rules PressureFit knows how to build a candidate for.
FETCH_RULES = (
    "packed-fifo",
    "packed-fit",
    "latest-safe",
    "demand",
)


@dataclass(frozen=True, slots=True)
class PressureFitOptions(OptionRecord):
    """PressureFit's own options: what it may try, and how hard.

    `PressureFit` takes one. The planner carries it into the plan key and
    never reads it; what every search is told is `GenericPlanningOptions`. A
    candidate is one resolved program searched with one residency strategy
    and one fetch rule, and every candidate the options name is searched.

    Parameters
    ----------
    resolution_options : tuple of Fraction, default DEFAULT_RESOLUTION_OPTIONS
        Which resolved programs to plan: each a share of the flexible
        task-alternative groups to recompute, as
        `validate_resolution_options` takes them. Kept distinct and
        ascending.
    tail_save_counts : tuple of int, default (0,)
        How many final eligible forward invocations of each forward/backward
        pass to save, as `validate_tail_save_counts` takes them. Each count
        is crossed with every share, which then applies to the remaining
        flexible groups; zero saves none. Kept distinct and ascending.
    residency_strategies : tuple of str, default ``("headroom-stall", "tight-stall")``
        How a candidate charges resident bytes and ranks the cuts at a
        boundary over capacity: ``"headroom-stall"``,
        ``"headroom-transfer"``, ``"tight-stall"`` or ``"tight-transfer"``.
        At least one, each once.
    fetch_rules : tuple of str, default FETCH_RULES
        At which task boundary a candidate starts each fetch:
        ``"packed-fifo"``, ``"packed-fit"``, ``"latest-safe"`` or
        ``"demand"``; all four by default. At least one, each once.
    max_repair_attempts : int, default 256
        Repairs one candidate may make before it answers with the best plan
        it reached. At least zero and below 2**32.
    capacity_refinement_bytes : int, default 268435456
        The most capacity a candidate gives back per round when its layout
        does not fit the pool, in bytes (256 MiB by default). A round gives
        back its overrun, or twice what the last round gave back when that
        round changed nothing, up to this bound. Zero removes the bound, so
        a round gives back at least the whole shortfall. At least zero and
        below 2**64.
    record_reduction_steps : bool, default False
        Record every plan each candidate held, as `CandidateDiagnostic.steps`.

    Raises
    ------
    ValueError
        If a strategy or fetch rule is unknown or repeated, or none is named; a
        share or count is invalid; ``max_repair_attempts`` is not a
        non-negative integer below 2**32; ``capacity_refinement_bytes`` is
        negative or not below 2**64; or ``record_reduction_steps`` is not a
        bool.
    """

    KIND: ClassVar[str] = "pressurefit"

    #: Which resolved programs PressureFit plans: the share of the flexible
    #: alternative groups to recompute, one resolved program per share, as
    #: exact fractions. Expanding a program into these is PressureFit's own
    #: business, which is why the share list is one of its options rather
    #: than the planner's.
    resolution_options: tuple[Fraction, ...] = DEFAULT_RESOLUTION_OPTIONS
    #: Counts of final flexible forward invocations to save in every pass.
    #: Every count crosses every resolution share; that share applies to the
    #: remaining flexible groups. (0,) saves no tail. Required saves do not
    #: consume the count.
    tail_save_counts: tuple[int, ...] = (0,)
    #: The strategies and rules a winner comes from. Naming more of the
    #: strategies or rules above widens the candidate set, and every
    #: candidate in it is searched.
    residency_strategies: tuple[str, ...] = (
        "headroom-stall",
        "tight-stall",
    )
    fetch_rules: tuple[str, ...] = (
        "packed-fifo",
        "packed-fit",
        "latest-safe",
        "demand",
    )
    #: How many monotonic repairs one candidate may make before it answers
    #: with the best plan it reached. Raising it buys quality where memory is
    #: tightest and costs planning time, which is what the workers pay for.
    max_repair_attempts: int = 256
    #: How much capacity a plan gives back at a time when its layout does
    #: not fit. The extent does not fall byte for byte with the capacity, so
    #: handing back the whole overage overshoots the capacity that would have
    #: fit and plans against a worse one. Stepping instead costs rounds and
    #: buys quality; zero hands back the whole shortfall, which converges in
    #: the fewest rounds and is the setting to reach for when planning time
    #: matters more than plan quality. A round that changed nothing hands
    #: back twice what the last did, within this bound, so an extent the cuts
    #: do not reach costs a few rounds rather than one per overrun.
    capacity_refinement_bytes: int = 256 * 1024 * 1024
    #: Record what each candidate's search actually did: one step per plan it
    #: held, with the objects the reducer cut to reach it and what became of
    #: it. Off by default because it costs an allocation per candidate that
    #: grows with the search -- worth paying to attribute planner time or
    #: explain a plan, and not worth paying in a sweep.
    record_reduction_steps: bool = False

    def __post_init__(self) -> None:
        # The planner reads it as a 64-bit count, which would wrap.
        if not 0 <= self.capacity_refinement_bytes < 1 << 64:
            raise ValueError("capacity_refinement_bytes is invalid")
        if not self.residency_strategies:
            raise ValueError("residency_strategies must not be empty")
        if not self.fetch_rules:
            raise ValueError("fetch_rules must not be empty")
        if len(set(self.residency_strategies)) != len(self.residency_strategies):
            raise ValueError("residency_strategies contains duplicates")
        if len(set(self.fetch_rules)) != len(self.fetch_rules):
            raise ValueError("fetch_rules contains duplicates")
        unknown_strategies = set(self.residency_strategies) - set(RESIDENCY_STRATEGIES)
        unknown_fetch = set(self.fetch_rules) - set(FETCH_RULES)
        if unknown_strategies:
            raise ValueError(
                f"unknown residency strategies: {sorted(unknown_strategies)}"
            )
        if unknown_fetch:
            raise ValueError(f"unknown fetch rules: {sorted(unknown_fetch)}")
        # The planner reads it as a 32-bit count, which would wrap.
        if (
            isinstance(self.max_repair_attempts, bool)
            or not isinstance(self.max_repair_attempts, int)
            or not 0 <= self.max_repair_attempts < 1 << 32
        ):
            raise ValueError(
                "max_repair_attempts must be a non-negative integer below 2**32"
            )
        if not isinstance(self.record_reduction_steps, bool):
            raise ValueError("record_reduction_steps must be a boolean")
        # Normalized here rather than at the point of use, so what the plan
        # key records is exactly what is planned.
        object.__setattr__(
            self,
            "resolution_options",
            validate_resolution_options(self.resolution_options),
        )
        object.__setattr__(
            self, "tail_save_counts", validate_tail_save_counts(self.tail_save_counts)
        )

    @classmethod
    def from_dict(cls, record: Mapping[str, object]) -> PressureFitOptions:
        """Read options back from what `OptionRecord.to_dict` wrote.

        Parameters
        ----------
        record : mapping of str to object
            Every option by name, and optionally ``kind``.

        Returns
        -------
        PressureFitOptions
            The options.

        Raises
        ------
        ValueError
            If ``record`` holds a key that is neither an option nor ``kind``,
            misses an option, or holds an invalid value.
        """
        unknown = set(record) - {"kind", *(field.name for field in fields(cls))}
        if unknown:
            raise ValueError(
                f"unsupported PressureFit options: {', '.join(sorted(unknown))}"
            )
        return super(PressureFitOptions, cls).from_dict(record)


__all__ = ["FETCH_RULES", "RESIDENCY_STRATEGIES", "PressureFitOptions"]
