"""The few numbers that describe a plan: the step it makes and where it goes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from shadowspill.planner.diagnostics import AlternativeCosts
from shadowspill.planner.result import ProgramPlanResult
from shadowspill.planner.search.toolkit.resolution import CostedAlternatives
from shadowspill.planner.serialization import _integer, _mapping


@dataclass(frozen=True, slots=True)
class PlanSummary:
    """What the selected plan promises, in one place.

    Only quantities that exist nowhere else on the report live here;
    budgets, transfer totals, and calibration stay on their canonical
    fields and profiles. The four time parts identify exactly:
    ``simulated_step_seconds == unconstrained_step_seconds +
    recomputation_overhead_seconds + idle_seconds +
    terminal_writeback_seconds``. The unconstrained step charges every
    task-alternative group its cheapest option and no waiting at all — the
    compute floor the geometry admits. A group counts as a recomputation
    selection when the option the search chose costs strictly more compute
    than that group's cheapest, whatever the options are named.
    """

    simulated_step_seconds: float
    unconstrained_step_seconds: float
    recomputation_overhead_seconds: float
    idle_seconds: float
    terminal_writeback_seconds: float
    recomputing_group_count: int
    task_alternative_group_count: int
    #: Groups that are a real decision. A resolution share is taken of these,
    #: so this is the denominator a recomputation count is against; the rest
    #: are forced, by structure or by their options keeping the same bytes.
    flexible_group_count: int
    #: Scheduled transfer traffic, summed from the simulation's transfer
    #: intervals, and the per-direction calibration the simulator planned
    #: against, from the result's simulation config: the coarsened rate and
    #: per-transfer latency each lane was priced with, which is what a reader
    #: comparing a plan against a measurement needs. Measured calibration
    #: lives on the report's transfer profiles; a result does not know it, and
    #: coarsening a profile here would answer for the wrong calibration
    #: whenever a plan came from the store.
    transfer_bytes_fetched: int = 0
    transfer_bytes_evicted: int = 0
    fetch_bandwidth_bytes_per_second: int = 0
    evict_bandwidth_bytes_per_second: int = 0
    fetch_latency_ns: int = 0
    evict_latency_ns: int = 0
    #: Wall time each frontend planning phase spent, in seconds, in phase
    #: order. A view over the report's ``phase_timings_ns``, which stays the
    #: stored record.
    planning_phase_seconds: Mapping[str, float] = field(
        default_factory=lambda: MappingProxyType({})
    )
    #: The candidate whose plan was selected: its residency strategy,
    #: fetch rule, coalescing, the repairs it had spent when it placed that
    #: plan (``repairs_at_best``), the fastest plan it could not place
    #: (``best_unplaced_makespan_ns``, ``unplaced_plans``) and
    #: ``placement_gap``, the answer over that plan.
    selected_candidate: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )

    @property
    def recomputing_group_fraction(self) -> float:
        if self.flexible_group_count == 0:
            return 0.0
        return self.recomputing_group_count / self.flexible_group_count

    def as_dict(self) -> dict[str, object]:
        return {
            "simulated_step_seconds": self.simulated_step_seconds,
            "unconstrained_step_seconds": self.unconstrained_step_seconds,
            "recomputation_overhead_seconds": self.recomputation_overhead_seconds,
            "idle_seconds": self.idle_seconds,
            "terminal_writeback_seconds": self.terminal_writeback_seconds,
            "recomputing_group_count": self.recomputing_group_count,
            "task_alternative_group_count": self.task_alternative_group_count,
            "flexible_group_count": self.flexible_group_count,
            "recomputing_group_fraction": self.recomputing_group_fraction,
            "transfer_bytes_fetched": self.transfer_bytes_fetched,
            "transfer_bytes_evicted": self.transfer_bytes_evicted,
            "fetch_bandwidth_bytes_per_second": (self.fetch_bandwidth_bytes_per_second),
            "evict_bandwidth_bytes_per_second": (self.evict_bandwidth_bytes_per_second),
            "fetch_latency_ns": self.fetch_latency_ns,
            "evict_latency_ns": self.evict_latency_ns,
            "planning_phase_seconds": dict(self.planning_phase_seconds),
            "selected_candidate": dict(self.selected_candidate),
        }

    @classmethod
    def from_dict(cls, value: object, path: str = "plan_summary") -> PlanSummary:
        """Read back what :meth:`as_dict` wrote.

        The inverse exists so a saved search is a record rather than a
        write-only log: a figure can be redrawn, or a run compared against an
        older one, without planning anything again. ``recomputing_group_fraction``
        is written for a reader's convenience and derived here, so it is ignored
        rather than trusted.
        """

        record = _mapping(value, path)

        def number(name: str, default: float = 0.0) -> float:
            item = record.get(name, default)
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError(f"{path}.{name}: expected a number")
            return float(item)

        def count(name: str, default: int = 0) -> int:
            return _integer(record.get(name, default), f"{path}.{name}")

        phases = _mapping(
            record.get("planning_phase_seconds", {}), f"{path}.planning_phase_seconds"
        )
        candidate = _mapping(
            record.get("selected_candidate", {}), f"{path}.selected_candidate"
        )
        return cls(
            simulated_step_seconds=number("simulated_step_seconds"),
            unconstrained_step_seconds=number("unconstrained_step_seconds"),
            recomputation_overhead_seconds=number("recomputation_overhead_seconds"),
            idle_seconds=number("idle_seconds"),
            terminal_writeback_seconds=number("terminal_writeback_seconds"),
            recomputing_group_count=count("recomputing_group_count"),
            task_alternative_group_count=count("task_alternative_group_count"),
            flexible_group_count=count("flexible_group_count"),
            transfer_bytes_fetched=count("transfer_bytes_fetched"),
            transfer_bytes_evicted=count("transfer_bytes_evicted"),
            fetch_bandwidth_bytes_per_second=count("fetch_bandwidth_bytes_per_second"),
            evict_bandwidth_bytes_per_second=count("evict_bandwidth_bytes_per_second"),
            fetch_latency_ns=count("fetch_latency_ns"),
            evict_latency_ns=count("evict_latency_ns"),
            planning_phase_seconds=MappingProxyType(
                {key: float(item) for key, item in phases.items()}
            ),
            selected_candidate=MappingProxyType(dict(candidate)),
        )


def summarize_selected_plan(
    result: ProgramPlanResult,
    *,
    phase_timings_ns: tuple[tuple[str, int], ...] = (),
) -> PlanSummary:
    """Derive one selected plan's :class:`PlanSummary` from its result."""

    program = result.program
    costs = AlternativeCosts.from_program(program)
    chosen = {item.group_id: item.option_id for item in result.selections}
    floor_ns = costs.floor_ns
    selected_ns = costs.selected_ns(chosen)
    span_ns = max(item.end_ns for item in result.simulation.task_intervals)
    makespan_ns = result.simulation.makespan_ns
    fetched = 0
    evicted = 0
    for interval in result.simulation.transfer_intervals:
        if interval.direction.value == "fetch":
            fetched += interval.bytes
        else:
            evicted += interval.bytes
    device = result.simulation_config.devices[0]
    selected: dict[str, object] = {}
    for problem in result.diagnostics.resolved_programs:
        # A policy is evaluated once per resolved program, so the selected
        # candidate is the one in the selected program: the same policy in
        # another program is a different evaluation with its own record.
        if problem.selection_id != result.diagnostics.selected_selection_id:
            continue
        # The plan to beat won: the answer is the plan the search was handed,
        # described by where it came from rather than by a candidate policy.
        if problem.incumbent is not None and problem.incumbent.selected:
            selected = {"incumbent": problem.incumbent.to_dict()}
        for candidate in problem.candidate_evaluations:
            if candidate.candidate_id == result.diagnostics.selected_candidate_id:
                selected = {
                    "residency_strategy": candidate.residency_strategy,
                    "fetch_rule": candidate.fetch_rule,
                    "coalesced": candidate.coalesced,
                    "repairs_at_best": candidate.repairs_at_best,
                    "best_unplaced_makespan_ns": candidate.best_unplaced_makespan_ns,
                    "unplaced_plans": candidate.unplaced_plans,
                    # The answer over the fastest plan that failed only
                    # placement: 1.0 means placing cost nothing.
                    "placement_gap": (
                        round(makespan_ns / candidate.best_unplaced_makespan_ns, 4)
                        if candidate.best_unplaced_makespan_ns
                        else None
                    ),
                }
    return PlanSummary(
        simulated_step_seconds=makespan_ns / 1e9,
        unconstrained_step_seconds=floor_ns / 1e9,
        recomputation_overhead_seconds=(selected_ns - floor_ns) / 1e9,
        idle_seconds=(span_ns - selected_ns) / 1e9,
        terminal_writeback_seconds=(makespan_ns - span_ns) / 1e9,
        recomputing_group_count=costs.recomputing(chosen),
        task_alternative_group_count=len(result.selections),
        flexible_group_count=CostedAlternatives.from_program(program).flexible_count,
        transfer_bytes_fetched=fetched,
        transfer_bytes_evicted=evicted,
        fetch_bandwidth_bytes_per_second=device.fetch_bandwidth_bytes_per_second,
        evict_bandwidth_bytes_per_second=device.evict_bandwidth_bytes_per_second,
        fetch_latency_ns=device.fetch_latency_ns,
        evict_latency_ns=device.evict_latency_ns,
        planning_phase_seconds=MappingProxyType(
            {name: duration / 1e9 for name, duration in phase_timings_ns}
        ),
        selected_candidate=MappingProxyType(selected),
    )
