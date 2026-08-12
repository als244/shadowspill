"""Convert residency spans into executable ordered memory actions."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

from shadowspill.ir import (
    MemoryAction,
    MemoryActionKind,
    MemoryLocation,
    MemorySchedule,
    ResidencySpec,
)
from shadowspill.simulator import SimulationConfig

from ._facts import PlanningFacts
from ._residency import ResidencyPlan, Span, _pressure_by_device
from .model import PressureFitInfeasibleError


@dataclass(frozen=True, slots=True)
class Reload:
    alias: int
    earliest_trigger: int
    latest_trigger: int
    entry_boundary: int


@dataclass(frozen=True, slots=True)
class Departure:
    alias: int
    trigger: int
    kind: MemoryActionKind


def _events_in_span(
    facts: PlanningFacts, alias: int, span: Span
) -> tuple[tuple[int, int], ...]:
    return tuple(
        event
        for event in facts.access_events[alias]
        if span.start <= event[0] <= span.end
    )


def _departure_task(facts: PlanningFacts, alias: int, span: Span) -> int:
    events = _events_in_span(facts, alias, span)
    if events:
        return max(task for _boundary, task in events)
    if not facts.tasks:
        raise PressureFitInfeasibleError(
            f"alias {facts.alias_ids[alias]!r} requires a pre-task departure, "
            "but the program has no task boundary",
            kind="missing_action_boundary",
        )
    return min(max(span.end, 0), len(facts.tasks) - 1)


def _entry_deadline(facts: PlanningFacts, alias: int, span: Span) -> int:
    events = _events_in_span(facts, alias, span)
    if events:
        first_task = min(task for _boundary, task in events)
        return first_task - 1
    return facts.last_boundary


def _has_write_since(
    facts: PlanningFacts,
    alias: int,
    refreshed_at: int,
    through: int,
) -> bool:
    return any(
        refreshed_at < boundary <= through for boundary in facts.write_boundaries[alias]
    )


def _transfer_runtime_ns(
    facts: PlanningFacts,
    config: SimulationConfig,
    alias: int,
) -> int:
    device_id = facts.alias_devices[alias]
    device = next(item for item in config.devices if item.device_id == device_id)
    size = facts.alias_sizes[alias]
    return (
        device.fetch_latency_ns
        + (size * 1_000_000_000 + device.fetch_bandwidth_bytes_per_second - 1)
        // device.fetch_bandwidth_bytes_per_second
    )


def _ideal_trigger_time(facts: PlanningFacts, trigger: int) -> int:
    if trigger < 0:
        return 0
    return facts.task_ideal_end_ns[trigger]


def _latest_trigger_at_or_before(
    facts: PlanningFacts,
    earliest: int,
    latest: int,
    target_time_ns: int,
) -> int:
    lower = max(earliest, 0)
    insertion = bisect_right(
        facts.task_ideal_end_ns,
        target_time_ns,
        lo=lower,
        hi=latest + 1,
    )
    return max(earliest, insertion - 1)


def _packed_triggers(
    facts: PlanningFacts,
    config: SimulationConfig,
    reloads: tuple[Reload, ...],
) -> dict[Reload, int]:
    selected: dict[Reload, int] = {}
    packed_start: dict[str, int] = {}
    for reload in sorted(
        reloads,
        key=lambda item: (item.latest_trigger, item.alias),
        reverse=True,
    ):
        device_id = facts.alias_devices[reload.alias]
        deadline = _ideal_trigger_time(facts, reload.latest_trigger)
        finish = min(deadline, packed_start.get(device_id, deadline))
        runtime = _transfer_runtime_ns(facts, config, reload.alias)
        desired_start = max(finish - runtime, 0)
        trigger = _latest_trigger_at_or_before(
            facts,
            reload.earliest_trigger,
            reload.latest_trigger,
            desired_start,
        )
        selected[reload] = trigger
        # A trigger is quantized to a task boundary, but the packed transfer
        # interval may begin later than that boundary.  Preserve that residual
        # lane occupancy when placing the next job; otherwise two transfers
        # are incorrectly modeled as overlapping on the single FETCH lane.
        packed_start[device_id] = max(
            _ideal_trigger_time(facts, trigger), desired_start
        )
    return selected


def _fit_clamped_triggers(
    facts: PlanningFacts,
    plan: ResidencyPlan,
    selected: dict[Reload, int],
    *,
    prefetch_headroom: bool,
) -> dict[Reload, int]:
    result = dict(selected)
    base_pressure = _pressure_by_device(
        facts,
        plan,
        prefetch_headroom=prefetch_headroom,
    )
    active_by_device = {
        device_id: [dict[Reload, None]() for _ in range(len(facts.tasks))]
        for device_id in facts.object_capacity_by_device
    }
    alias_counts = {
        device_id: [dict[int, int]() for _ in range(len(facts.tasks))]
        for device_id in facts.object_capacity_by_device
    }
    used = {device_id: list(values) for device_id, values in base_pressure.items()}

    def add_active(reload: Reload, trigger: int) -> None:
        device_id = facts.alias_devices[reload.alias]
        for boundary in range(max(trigger, 0), reload.entry_boundary):
            if plan.resident(reload.alias, boundary):
                continue
            active_by_device[device_id][boundary][reload] = None
            counts = alias_counts[device_id][boundary]
            previous = counts.get(reload.alias, 0)
            counts[reload.alias] = previous + 1
            if previous == 0:
                used[device_id][boundary + 1] += facts.alias_sizes[reload.alias]

    def move_active(reload: Reload, old_trigger: int, new_trigger: int) -> None:
        device_id = facts.alias_devices[reload.alias]
        for boundary in range(max(old_trigger, 0), max(new_trigger, 0)):
            if plan.resident(reload.alias, boundary):
                continue
            active_by_device[device_id][boundary].pop(reload)
            counts = alias_counts[device_id][boundary]
            remaining = counts[reload.alias] - 1
            if remaining == 0:
                del counts[reload.alias]
                used[device_id][boundary + 1] -= facts.alias_sizes[reload.alias]
            else:
                counts[reload.alias] = remaining

    for reload, trigger in result.items():
        add_active(reload, trigger)

    while True:
        changed = False
        for device_id, capacity in facts.object_capacity_by_device.items():
            for boundary in range(0, facts.last_boundary + 1):
                if used[device_id][boundary + 1] <= capacity:
                    continue
                movable = [
                    reload
                    for reload in active_by_device[device_id][boundary]
                    if result[reload] < reload.latest_trigger
                ]
                if not movable:
                    continue
                chosen = max(
                    movable,
                    key=lambda reload: (
                        reload.entry_boundary,
                        facts.alias_sizes[reload.alias],
                        reload.alias,
                    ),
                )
                old_trigger = result[chosen]
                new_trigger = min(boundary + 1, chosen.latest_trigger)
                result[chosen] = new_trigger
                move_active(chosen, old_trigger, new_trigger)
                changed = True
                break
            if changed:
                break
        if not changed:
            return result


def choose_prefetch_triggers(
    facts: PlanningFacts,
    config: SimulationConfig,
    plan: ResidencyPlan,
    reloads: tuple[Reload, ...],
    rule: str,
    *,
    prefetch_headroom: bool,
) -> dict[Reload, int]:
    if rule == "latest-safe":
        return {item: item.latest_trigger for item in reloads}
    selected = _packed_triggers(facts, config, reloads)
    if rule == "packed-fit":
        return _fit_clamped_triggers(
            facts,
            plan,
            selected,
            prefetch_headroom=prefetch_headroom,
        )
    return selected


def _initial_schedule(
    facts: PlanningFacts, plan: ResidencyPlan
) -> tuple[ResidencySpec, ...]:
    values: list[ResidencySpec] = []
    for alias, location in enumerate(facts.initial_locations):
        if location is None or facts.alias_sizes[alias] == 0:
            continue
        selected = (
            MemoryLocation.DEVICE if plan.resident(alias, -1) else MemoryLocation.HOST
        )
        values.append(ResidencySpec(facts.alias_ids[alias], selected))
    return tuple(values)


def emit_schedule(
    facts: PlanningFacts,
    config: SimulationConfig,
    plan: ResidencyPlan,
    prefetch_rule: str,
    *,
    coalesced: bool,
    prefetch_headroom: bool = False,
) -> MemorySchedule:
    """Emit one immutable schedule without changing any task boundary."""

    departures: list[Departure] = []
    reloads: list[Reload] = []
    for alias, spans in enumerate(plan.spans):
        if not spans or facts.alias_sizes[alias] == 0:
            continue
        host_refreshed = (
            -1
            if (
                facts.initial_locations[alias] is MemoryLocation.HOST
                or facts.alias_retain_spill_copy[alias]
            )
            else -2
        )
        previous_departure: Departure | None = None
        for span_index, span in enumerate(spans):
            produced_at_entry = span.start in facts.production_boundaries[alias]
            if span.start > -1 and not produced_at_entry:
                latest = _entry_deadline(facts, alias, span)
                earliest = 0
                if previous_departure is not None:
                    earliest = previous_departure.trigger + 1
                    if (
                        coalesced
                        and previous_departure.kind is MemoryActionKind.RELEASE
                    ):
                        earliest = previous_departure.trigger
                if latest < earliest:
                    raise PressureFitInfeasibleError(
                        f"alias {facts.alias_ids[alias]!r} has no legal prefetch "
                        f"boundary between tasks {earliest} and {latest}",
                        kind="prefetch_window",
                        device_id=facts.alias_devices[alias],
                    )
                reloads.append(Reload(alias, earliest, latest, span.start))

            departure_task = _departure_task(facts, alias, span)
            has_later_span = span_index + 1 < len(spans)
            final = facts.final_locations[alias]
            if has_later_span:
                if facts.alias_retain_spill_copy[alias] and not _has_write_since(
                    facts, alias, host_refreshed, span.end
                ):
                    kind = MemoryActionKind.RELEASE
                else:
                    kind = MemoryActionKind.OFFLOAD
                    host_refreshed = span.end
                previous_departure = Departure(alias, departure_task, kind)
                departures.append(previous_departure)
            elif final is MemoryLocation.DEVICE:
                continue
            elif final is MemoryLocation.HOST:
                if facts.alias_retain_spill_copy[alias] and not _has_write_since(
                    facts, alias, host_refreshed, span.end
                ):
                    kind = MemoryActionKind.RELEASE
                else:
                    kind = MemoryActionKind.OFFLOAD
                previous_departure = Departure(alias, departure_task, kind)
                departures.append(previous_departure)
            else:
                previous_departure = Departure(
                    alias, departure_task, MemoryActionKind.RELEASE
                )
                departures.append(previous_departure)

    triggers = choose_prefetch_triggers(
        facts,
        config,
        plan,
        tuple(reloads),
        prefetch_rule,
        prefetch_headroom=prefetch_headroom,
    )
    actions_by_task: dict[int, list[tuple[int, MemoryActionKind]]] = {}
    for departure in departures:
        actions_by_task.setdefault(departure.trigger, []).append(
            (departure.alias, departure.kind)
        )
    for reload in reloads:
        actions_by_task.setdefault(triggers[reload], []).append(
            (reload.alias, MemoryActionKind.PREFETCH)
        )
    kind_order = {
        MemoryActionKind.RELEASE: 0,
        MemoryActionKind.OFFLOAD: 1,
        MemoryActionKind.PREFETCH: 2,
    }
    actions = tuple(
        MemoryAction(
            facts.tasks[task].task_id,
            facts.alias_ids[alias],
            kind,
        )
        for task in sorted(actions_by_task)
        for alias, kind in sorted(
            actions_by_task[task], key=lambda item: (kind_order[item[1]], item[0])
        )
    )
    if coalesced:
        release_keys = {
            (action.trigger_task_id, action.alias_group_id)
            for action in actions
            if action.kind is MemoryActionKind.RELEASE
        }
        prefetch_keys = {
            (action.trigger_task_id, action.alias_group_id)
            for action in actions
            if action.kind is MemoryActionKind.PREFETCH
        }
        coalesced_keys = release_keys & prefetch_keys
        actions = tuple(
            action
            for action in actions
            if (action.trigger_task_id, action.alias_group_id) not in coalesced_keys
        )
    schedule = MemorySchedule(
        initial_residency=_initial_schedule(facts, plan),
        actions=actions,
        final_residency=tuple(
            ResidencySpec(facts.alias_ids[alias], location)
            for alias, location in enumerate(facts.final_locations)
            if location is not None and facts.alias_sizes[alias] != 0
        ),
    )
    schedule._validate_selected(facts.program, facts.tasks)
    return schedule


__all__ = ["emit_schedule"]
