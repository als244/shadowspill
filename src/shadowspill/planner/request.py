"""What a caller asks PressureFit for."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from enum import Enum, StrEnum
from typing import Any


class InitialPlacement(StrEnum):
    """How host-origin objects may be placed before the first task."""

    REQUIRED = "required"
    GREEDY = "greedy"


@dataclass(frozen=True, slots=True)
class PressureFitOptions:
    """Bounded heuristic candidate configuration.

    Every field here is part of a planned program's identity, worker count
    included. Worker count enters no candidate's identity and breaks no tie
    between two candidates, but candidates measure a layout only when the
    shared best-placed record says it could win, so which worker places
    first decides which candidates are ever measured, and two searches over
    the same problem at different worker counts can answer with different
    plans. Zero selects all available logical CPUs; one forces serial
    evaluation.

    ``deterministic`` is how a search reproduces without giving up its
    workers: the placement gate consults only the candidate's own placed
    plans instead of the shared record, so every outcome is a pure function
    of that candidate's inputs and any worker count answers the same. It
    costs wall time, because the shared bound is what lets a candidate skip
    measuring a plan that cannot win.
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
    fetch_rules: tuple[str, ...] = (
        "packed-fifo",
        "packed-fit",
        "latest-safe",
        "demand",
    )
    evaluate_coalesced: bool = True
    #: How many monotonic repairs one candidate may make before it answers
    #: with the best plan it reached. Measured over the 2,520-point corpus,
    #: 256 changes no candidate's status against 64 and improves the mean
    #: makespan by 0.40%, with the wins concentrated where memory is
    #: tightest; it costs planning time, which the workers are what pays for.
    max_repair_attempts: int = 256
    #: How much capacity a plan gives back at a time when its layout does
    #: not fit. The extent does not fall byte for byte with the capacity, so
    #: handing back the whole overage overshoots the capacity that would have
    #: fit, and the plan built below that capacity is materially worse than
    #: the one just under the line. Stepping instead costs rounds and buys
    #: quality: across a 45-point slice, stepping here rather than handing
    #: back the shortfall moved the median 0.4 points and the worst point
    #: 1.2, for about 40% more planning time. Zero hands back the whole
    #: shortfall, which converges in the fewest rounds and is the setting to
    #: reach for when planning time matters more than the last percent.
    capacity_refinement_bytes: int = 256 * 1024 * 1024
    #: Record what each candidate's search actually did: one step per plan it
    #: held, with the objects the reducer cut to reach it and what became of
    #: it. Off by default because it costs an allocation per candidate that
    #: grows with the search -- worth paying to attribute planner time or
    #: explain a plan, and not worth paying in a sweep.
    record_reduction_steps: bool = False
    workers: int = 0
    #: Make every candidate's outcome a pure function of its inputs, so
    #: parallel planning reproduces exactly run to run. The placement gate
    #: then consults only the candidate's own placed plans, never the shared
    #: best-placed record, which costs additional placement measurements.
    #: Off by default: the shared gate is faster, and the default search is
    #: stable without being bit-reproducible.
    deterministic: bool = False
    #: Objects smaller than this many bytes are not eligible to be evicted
    #: mid-step: they stay resident from their first to their last access,
    #: and the planner charges them at every boundary in between. Their
    #: boundary contract is untouched -- an opening fetch, a release after
    #: the last access, a terminal writeback when modified. The default is
    #: 1 MiB, because a copy under that size is latency-bound and its bytes
    #: hardly relieve a boundary, while every such object is a cut
    #: candidate, a dispatch, and an event. Zero makes every object
    #: eligible, which is what a caller planning byte-sized objects wants.
    minimum_object_bytes_evict_eligible: int = 1 << 20

    def to_dict(self) -> dict[str, Any]:
        """Every option, in declaration order, as JSON-compatible values.

        Derived from the dataclass rather than a written-out list, so an
        option added later is carried by this and by everything built on
        it without a second edit.
        """

        record: dict[str, Any] = {}
        for option in fields(self):
            value = getattr(self, option.name)
            if isinstance(value, Enum):
                value = value.value
            elif isinstance(value, tuple):
                value = list(value)
            record[option.name] = value
        return record

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> PressureFitOptions:
        """Rebuild what :meth:`to_dict` wrote. Every option must be present.

        A record missing an option is a record from a different version of
        this type, and reading it as though the absent options held their
        current defaults is how a replay silently plans a different
        problem than the run it replays.
        """

        missing = sorted(
            option.name for option in fields(cls) if option.name not in record
        )
        if missing:
            raise ValueError(f"options omit {', '.join(missing)}")
        values: dict[str, Any] = {}
        for option in fields(cls):
            value = record[option.name]
            if isinstance(option.default, Enum):
                value = type(option.default)(value)
            elif isinstance(option.default, tuple):
                value = tuple(value)
            values[option.name] = value
        return cls(**values)

    def __post_init__(self) -> None:
        if self.capacity_refinement_bytes < 0:
            raise ValueError("capacity_refinement_bytes is invalid")
        if self.minimum_object_bytes_evict_eligible < 0:
            raise ValueError("minimum_object_bytes_evict_eligible is invalid")
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
        known_strategies = {
            "headroom-stall",
            "headroom-transfer",
            "tight-stall",
            "tight-transfer",
            "relaxed-stall",
        }
        known_fetch = {
            "packed-fifo",
            "packed-fit",
            "interval-entry",
            "latest-safe",
            "demand",
        }
        unknown_strategies = set(self.residency_strategies) - known_strategies
        unknown_fetch = set(self.fetch_rules) - known_fetch
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
        if (
            isinstance(self.workers, bool)
            or not isinstance(self.workers, int)
            or self.workers < 0
        ):
            raise ValueError("workers must be a non-negative integer")


__all__ = ["InitialPlacement", "PressureFitOptions"]
