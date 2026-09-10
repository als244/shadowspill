"""What PressureFit is told: its candidate space and how hard it tries."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import ClassVar

from ....request import InitialPlacement, OptionRecord
from ...toolkit.resolution import (
    DEFAULT_RESOLUTION_OPTIONS,
    validate_resolution_options,
)

#: The residency strategies PressureFit knows how to build a candidate for.
RESIDENCY_STRATEGIES = (
    "headroom-stall",
    "headroom-transfer",
    "tight-stall",
    "tight-transfer",
    "relaxed-stall",
)
#: The fetch-trigger rules PressureFit knows how to build a candidate for.
FETCH_RULES = (
    "packed-fifo",
    "packed-fit",
    "interval-entry",
    "latest-safe",
    "demand",
)


@dataclass(frozen=True, slots=True)
class PressureFitOptions(OptionRecord):
    """PressureFit's own controls: what it may try, and how hard.

    The planner carries this into the plan key and never reads it, so the
    candidate space can grow without the planner learning a new word. What
    every search is told instead -- workers, determinism, which objects may
    move -- is :class:`~shadowspill.planner.SearchOptions`.
    """

    KIND: ClassVar[str] = "pressurefit"

    initial_placement: InitialPlacement = InitialPlacement.GREEDY
    #: Which resolved programs PressureFit plans: the share of the flexible
    #: alternative groups to recompute, one resolved program per share, as
    #: exact fractions. Expanding a program into these is PressureFit's own
    #: business, which is why the share list is one of its options rather
    #: than the planner's.
    resolution_options: tuple[Fraction, ...] = DEFAULT_RESOLUTION_OPTIONS
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
    #: Try a coalesced variant of every candidate as well, which doubles the
    #: candidate set.
    evaluate_coalesced: bool = True
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
    #: matters more than plan quality.
    capacity_refinement_bytes: int = 256 * 1024 * 1024
    #: Record what each candidate's search actually did: one step per plan it
    #: held, with the objects the reducer cut to reach it and what became of
    #: it. Off by default because it costs an allocation per candidate that
    #: grows with the search -- worth paying to attribute planner time or
    #: explain a plan, and not worth paying in a sweep.
    record_reduction_steps: bool = False
    #: Let a plan that has simulated split an eviction something waited on:
    #: a write-back at the boundary where the value was last written, and a
    #: release where the eviction was. The plan is simulated again and the
    #: split kept only if it got faster, so this widens what the search may
    #: consider rather than deciding anything. Off by default: it costs a
    #: second simulation per split.
    split_write_backs: bool = False

    def __post_init__(self) -> None:
        if self.capacity_refinement_bytes < 0:
            raise ValueError("capacity_refinement_bytes is invalid")
        if not isinstance(self.initial_placement, InitialPlacement):
            raise ValueError("initial_placement is invalid")
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
        if (
            isinstance(self.max_repair_attempts, bool)
            or not isinstance(self.max_repair_attempts, int)
            or self.max_repair_attempts < 0
        ):
            raise ValueError("max_repair_attempts must be a non-negative integer")
        if not isinstance(self.record_reduction_steps, bool):
            raise ValueError("record_reduction_steps must be a boolean")
        if not isinstance(self.evaluate_coalesced, bool):
            raise ValueError("evaluate_coalesced must be a boolean")
        if not isinstance(self.split_write_backs, bool):
            raise ValueError("split_write_backs must be a boolean")
        # Normalized here rather than at the point of use, so what the plan
        # key records is exactly what is planned.
        object.__setattr__(
            self,
            "resolution_options",
            validate_resolution_options(self.resolution_options),
        )


__all__ = ["FETCH_RULES", "RESIDENCY_STRATEGIES", "PressureFitOptions"]
