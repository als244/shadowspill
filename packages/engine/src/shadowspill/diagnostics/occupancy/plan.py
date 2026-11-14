"""Plan for pool-occupancy diagnostics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

from shadowspill.diagnostics.transfer_phases import transfer_phase_data

from .clock import Clock, _busy_ns
from .execution import LeaseClock, execution_occupancy, placement_rows, task_spans
from .facts import ProgramFacts
from .intervals import (
    WORKSPACE,
    Interval,
    ObjectFacts,
    Occupancy,
    TaskSpan,
    TransferSpan,
)
from .spill import _interval, spill_occupancy


@dataclass(frozen=True, slots=True)
class PlanOccupancy:
    """One view of a plan: its pools, its lanes, and the clock they are on."""

    view: str
    spill: Occupancy
    execution: Occupancy | None
    tasks: tuple[TaskSpan, ...]
    transfers: tuple[TransferSpan, ...]
    facts: ProgramFacts
    clock: Clock
    #: What the plan was priced against: the simulation config's fetch and
    #: evict bandwidths (bytes per second) and latencies (ns), by lane.
    assumed: Mapping[str, float] = field(default_factory=dict)
    #: On a traced view, how many lease instants no traced event named, so
    #: they were interpolated between task boundaries.
    interpolated_leases: int = 0
    #: Independent NCCL intervals, already included in their parent task or
    #: transfer. Preserve capture completeness so missing data stays explicit.
    collectives: Mapping[str, Any] | None = None
    transfer_phases: Mapping[str, Any] | None = None

    @property
    def untimed_transfers(self) -> int:
        """Simulated transfers the trace could not time, so they are off the lanes."""

        return self.clock.untimed_transfers

    @property
    def pools(self) -> tuple[Occupancy, ...]:
        """The execution pool, when the plan has a layout, then the spill
        pool: the order the page stacks them."""

        return ((self.execution,) if self.execution is not None else ()) + (self.spill,)

    @property
    def clock_label(self) -> str:
        if self.view == "traced":
            return "device clock (traced step)"
        if self.view == "unconstrained":
            return "unconstrained: every object resident, tasks at their profiled floor"
        if self.view == "all_save":
            if self.facts.recompute_tasks:
                return (
                    "cheapest available profiles: includes recompute; "
                    "this is not a measured all-save baseline"
                )
            return (
                "all save: every alternative at its cheapest, every object resident,"
                " tasks at their profiled floor"
            )
        return "simulated clock"


def attribute(
    selection: Mapping[str, Any],
    program: Mapping[str, Any],
    *,
    diagnostics: Mapping[str, Any] | None = None,
) -> PlanOccupancy:
    """Both pools and the three lanes of a stored plan, on the simulated clock
    or, with a traced step's diagnostics, on the device's."""

    if "simulation_result" not in selection:
        raise ValueError(
            "this selection holds no simulation result: the store names a plan"
            " here without its evidence, so there is nothing to walk"
        )
    facts = ProgramFacts.build(program, selection.get("selections", ()))
    simulation = selection["simulation_result"]
    clock = (
        Clock.simulated(simulation)
        if diagnostics is None
        else Clock.traced(simulation, diagnostics)
    )
    config = selection.get("simulation") or {}
    spill, transfers = spill_occupancy(
        facts,
        selection["schedule"],
        clock,
        reported_peak_bytes=int(simulation["spill_peak_bytes"]),
        capacity_bytes=(
            int(config["spill_capacity_bytes"])
            if "spill_capacity_bytes" in config
            else None
        ),
    )
    execution = None
    placed = None
    layout = selection.get("admission_certificate", {}).get("layout")
    if layout is not None:
        layout = {**layout, "placements": placement_rows(layout)}
        peaks = simulation.get("device_peaks") or ()
        reported = int(peaks[0]["total_bytes"]) if peaks else None
        if diagnostics is not None:
            placed = LeaseClock.build(
                simulation, selection["schedule"], Clock.simulated(simulation), clock
            )
        execution = execution_occupancy(
            facts, layout, clock, placed=placed, reported_peak_bytes=reported
        )
    device = (config.get("devices") or [{}])[0]
    assumed = {
        key: float(device[key])
        for key in (
            "fetch_solo_bandwidth_bytes_per_second",
            "fetch_concurrent_bandwidth_bytes_per_second",
            "evict_solo_bandwidth_bytes_per_second",
            "evict_concurrent_bandwidth_bytes_per_second",
            "fetch_latency_ns",
            "evict_latency_ns",
        )
        if device.get(key) is not None
    }
    # The planned blend always comes from the simulation, even on a measured
    # page. Never price the trace with its own achieved rate.
    for direction in ("fetch", "evict"):
        copies = [
            t for t in simulation["transfer_intervals"] if t["direction"] == direction
        ]
        busy = _busy_ns((t["start_ns"], t["end_ns"]) for t in copies)
        if busy:
            assumed[direction + "_blended_bandwidth_bytes_per_second"] = (
                sum(facts.alias_size[t["alias_group_id"]] for t in copies) * 1e9 / busy
            )
    return PlanOccupancy(
        "traced" if diagnostics is not None else "simulated",
        spill,
        execution,
        task_spans(facts, clock),
        transfers,
        facts,
        clock,
        assumed,
        0 if placed is None else placed.interpolated,
        None if diagnostics is None else diagnostics.get("collectives"),
        None if diagnostics is None else transfer_phase_data(program, diagnostics),
    )


def cheapest_selections(program: Mapping[str, Any]) -> list[dict[str, str]]:
    """Every alternative group at its cheapest option by the program's
    profiles -- the selection beneath the planner's compute floor -- in the
    form a stored plan records its ``selections``."""

    runtime = {
        profile["profile_id"]: int(profile.get("runtime_ns", 0))
        for profile in program.get("profiles", ())
    }
    task_ns = {
        task["task_id"]: runtime.get(str(task.get("profile_id")), 0)
        for task in program["tasks"]
    }
    chosen: list[dict[str, str]] = []
    for group in program.get("task_alternative_groups", ()):
        cheapest = min(
            group["options"],
            key=lambda option: sum(
                task_ns.get(task_id, 0) for task_id in option["active_task_ids"]
            ),
        )
        chosen.append(
            {"group_id": group["group_id"], "option_id": cheapest["option_id"]}
        )
    return chosen


def unconstrained(
    program: Mapping[str, Any], selections: Iterable[Mapping[str, str]]
) -> PlanOccupancy:
    """A plan's own floor: its alternatives as ``selections`` fixes them,
    with nothing else to plan around. Every task back to back at its
    profiled time, every object resident from its production, or from the
    step's start for checkpoint state, to its last use, and each task's
    profiled workspace while it runs. Nothing spills and the lanes are
    empty; the execution pool's peak is what the plan would need to run
    this way."""

    return _floor(program, list(selections), "unconstrained")


def all_save(program: Mapping[str, Any]) -> PlanOccupancy:
    """The program's floor with every alternative at its cheapest option --
    save, for a save-or-recompute choice -- which is the step the planner
    reports as unconstrained. The same walk as `unconstrained` otherwise."""

    return _floor(program, cheapest_selections(program), "all_save")


def _floor(
    program: Mapping[str, Any], selections: Sequence[Mapping[str, str]], view: str
) -> PlanOccupancy:
    facts = ProgramFacts.build(program, selections)
    tasks = {task["task_id"]: task for task in program["tasks"]}
    profiles = {
        profile["profile_id"]: profile for profile in program.get("profiles", ())
    }
    task_start: dict[str, int] = {}
    task_end: dict[str, int] = {}
    intervals: list[Interval] = []
    now = 0
    for task_id in facts.task_order:
        profile = profiles.get(str(tasks[task_id].get("profile_id")), {})
        task_start[task_id] = now
        now += int(profile.get("runtime_ns", 0))
        task_end[task_id] = now
        scratch = int(profile.get("workspace_bytes", 0))
        if scratch > 0 and now > task_start[task_id]:
            intervals.append(
                Interval(task_start[task_id], now, scratch, None, WORKSPACE, WORKSPACE)
            )
    clock = Clock(task_start, task_end, {}, now, {})
    by_alias: dict[str, list[ObjectFacts]] = defaultdict(list)
    for item in facts.objects.values():
        by_alias[item.alias_group_id].append(item)
    for alias_id, members in by_alias.items():
        size = facts.alias_size.get(alias_id, 0)
        if size == 0:
            continue
        # state the step keeps is resident throughout; a step object holds its
        # slot from its first production to its last use, an output to the end
        checkpoint = any(item.persistence != "step" for item in members)
        starts = sorted(
            {
                task_start[item.producer]
                for item in members
                if item.producer is not None and item.producer in task_start
            }
        )
        opens = 0 if checkpoint or not starts else starts[0]
        uses = [
            task_end[task_id]
            for item in members
            for task_id in item.consumers
            if task_id in task_end
        ] + [
            task_end[item.producer]
            for item in members
            if item.producer is not None and item.producer in task_end
        ]
        closes = (
            now
            if checkpoint or any(item.role == "output" for item in members)
            else max(uses, default=now)
        )
        edges = [opens, *(start for start in starts if opens < start < closes), closes]
        for begin, end in pairwise(edges):
            if end > begin:
                intervals.append(_interval(facts, clock, alias_id, begin, end, size))
    return PlanOccupancy(
        view,
        Occupancy("spill", ()),
        Occupancy("execution", tuple(intervals)),
        task_spans(facts, clock),
        (),
        facts,
        clock,
        {},
    )
