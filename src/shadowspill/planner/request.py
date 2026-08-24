"""What a caller asks PressureFit for."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class InitialPlacement(StrEnum):
    """How host-origin objects may be placed before the first task."""

    REQUIRED = "required"
    GREEDY = "greedy"


@dataclass(frozen=True, slots=True)
class PressureFitOptions:
    """Bounded heuristic candidate configuration.

    Worker count changes only evaluation concurrency. It never enters candidate
    identity or tie-breaking. Zero selects all available logical CPUs; one
    forces serial evaluation.
    """

    initial_placement: InitialPlacement = InitialPlacement.GREEDY
    # relaxed-stall (byte-identical to tight-stall) and interval-entry
    # never carry a winner: a 435-point regression replay reproduced
    # every schedule digest exactly without them at ~1.25x less
    # candidate set work. Both remain valid explicit options.
    residency_strategies: tuple[str, ...] = (
        "headroom-stall",
        "headroom-transfer",
        "tight-stall",
        "tight-transfer",
    )
    prefetch_rules: tuple[str, ...] = (
        "packed-fifo",
        "packed-fit",
        "latest-safe",
        "demand",
    )
    evaluate_coalesced: bool = True
    max_repair_attempts: int = 64
    #: Keep repairing a candidate whose plan already simulates while it still
    #: comes up short of capacity somewhere, returning its best plan by
    #: makespan rather than the first one that worked. A plan that waits for
    #: memory is valid but not finished, and the wait is time it pays.
    repair_while_stalling: bool = False
    #: Candidate policies to skip, named as `strategy/rule` with a
    #: `-coalesced` suffix -- the same form `CandidateDiagnostic.candidate_id`
    #: reports. A plan that cannot be placed physically is a fact about the
    #: policy that produced it, so a caller can rule that policy out and ask
    #: for another plan at the same capacity instead of reducing capacity for
    #: every policy at once.
    excluded_candidates: tuple[str, ...] = ()
    workers: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.initial_placement, InitialPlacement):
            raise ValueError("initial_placement is invalid")
        if not self.residency_strategies:
            raise ValueError("residency_strategies must not be empty")
        if not self.prefetch_rules:
            raise ValueError("prefetch_rules must not be empty")
        if len(set(self.residency_strategies)) != len(self.residency_strategies):
            raise ValueError("residency_strategies contains duplicates")
        if len(set(self.prefetch_rules)) != len(self.prefetch_rules):
            raise ValueError("prefetch_rules contains duplicates")
        known_strategies = {
            "headroom-stall",
            "headroom-transfer",
            "tight-stall",
            "tight-transfer",
            "relaxed-stall",
        }
        known_prefetch = {
            "packed-fifo",
            "packed-fit",
            "interval-entry",
            "latest-safe",
            "demand",
        }
        unknown_strategies = set(self.residency_strategies) - known_strategies
        unknown_prefetch = set(self.prefetch_rules) - known_prefetch
        if unknown_strategies:
            raise ValueError(
                f"unknown residency strategies: {sorted(unknown_strategies)}"
            )
        if unknown_prefetch:
            raise ValueError(f"unknown prefetch rules: {sorted(unknown_prefetch)}")
        if (
            isinstance(self.max_repair_attempts, bool)
            or not isinstance(self.max_repair_attempts, int)
            or self.max_repair_attempts < 0
        ):
            raise ValueError("max_repair_attempts must be a non-negative integer")
        if (
            isinstance(self.workers, bool)
            or not isinstance(self.workers, int)
            or self.workers < 0
        ):
            raise ValueError("workers must be a non-negative integer")


__all__ = ["InitialPlacement", "PressureFitOptions"]
