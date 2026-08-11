"""Deterministic interval construction and pressure reduction."""

from __future__ import annotations

from dataclasses import dataclass

from shadowspill.ir import MemoryLocation
from shadowspill.simulator import SimulationConfig

from ._facts import PlanningFacts
from .model import InitialPlacement, PressureFitInfeasibleError


@dataclass(frozen=True, order=True, slots=True)
class Span:
    """Inclusive task-boundary residency span."""

    start: int
    end: int

    def contains(self, boundary: int) -> bool:
        return self.start <= boundary <= self.end


@dataclass(frozen=True, slots=True)
class ResidencyPlan:
    spans: tuple[tuple[Span, ...], ...]
    anchors: tuple[frozenset[int], ...]

    def resident(self, alias: int, boundary: int) -> bool:
        return any(span.contains(boundary) for span in self.spans[alias])


@dataclass(frozen=True, slots=True)
class Cut:
    alias: int
    span_index: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return max(self.end - self.start + 1, 0)


def _seed_from_anchors(anchors: tuple[frozenset[int], ...]) -> ResidencyPlan:
    spans: list[tuple[Span, ...]] = []
    for values in anchors:
        if not values:
            spans.append(())
        else:
            spans.append((Span(min(values), max(values)),))
    return ResidencyPlan(tuple(spans), anchors)


def boundary_bytes(
    facts: PlanningFacts,
    plan: ResidencyPlan,
    boundary: int,
    device_id: str,
    *,
    prefetch_headroom: bool = False,
) -> int:
    def charged(alias: int) -> bool:
        for span in plan.spans[alias]:
            effective_start = span.start
            if (
                prefetch_headroom
                and span.start > -1
                and span.start not in facts.production_boundaries[alias]
            ):
                effective_start -= 1
            if not (effective_start <= boundary <= span.end):
                continue
            if boundary < 0 or span.end != boundary:
                return True
            if facts.final_locations[alias] is MemoryLocation.DEVICE:
                return True
            # A value whose final access is the task that just completed can
            # depart at this boundary before the next task is admitted. Its
            # output allocation was already charged at that producing task.
            return any(
                task > boundary
                for anchor, task in facts.access_events[alias]
                if span.start <= anchor <= span.end
            )
        return False

    resident = sum(
        facts.alias_sizes[alias]
        for alias in range(len(facts.alias_ids))
        if facts.alias_devices[alias] == device_id and charged(alias)
    )
    task = boundary + 1
    if task < 0 or task >= len(facts.tasks):
        return resident
    reservations = sum(
        facts.alias_sizes[alias]
        for alias in facts.output_reservations[task]
        if facts.alias_devices[alias] == device_id and not charged(alias)
    )
    return resident + reservations


def pressure_by_boundary(
    facts: PlanningFacts, plan: ResidencyPlan
) -> tuple[dict[str, int], ...]:
    pressure = _pressure_by_device(facts, plan)
    return tuple(
        {
            device_id: pressure[device_id][boundary + 1]
            for device_id in facts.object_capacity_by_device
        }
        for boundary in range(-1, facts.last_boundary + 1)
    )


def _charged_span(
    facts: PlanningFacts,
    alias: int,
    span: Span,
    *,
    prefetch_headroom: bool,
) -> Span | None:
    start = span.start
    if (
        prefetch_headroom
        and start > -1
        and start not in facts.production_boundaries[alias]
    ):
        start -= 1
    end = span.end
    if end >= 0 and facts.final_locations[alias] is not MemoryLocation.DEVICE:
        has_future_access = any(
            task > end
            for anchor, task in facts.access_events[alias]
            if span.start <= anchor <= span.end
        )
        if not has_future_access:
            end -= 1
    return None if end < start else Span(start, end)


def _pressure_by_device(
    facts: PlanningFacts,
    plan: ResidencyPlan,
    *,
    prefetch_headroom: bool = False,
) -> dict[str, tuple[int, ...]]:
    """Sweep all charged spans once instead of scanning aliases per boundary."""

    boundary_count = len(facts.tasks) + 1
    differences = {
        device_id: [0] * (boundary_count + 1)
        for device_id in facts.object_capacity_by_device
    }
    charged_spans: list[tuple[Span, ...]] = []
    for alias, spans in enumerate(plan.spans):
        charged_values: list[Span] = []
        for span in spans:
            interval = _charged_span(
                facts,
                alias,
                span,
                prefetch_headroom=prefetch_headroom,
            )
            if interval is not None:
                charged_values.append(interval)
        charged = tuple(charged_values)
        charged_spans.append(charged)
        difference = differences[facts.alias_devices[alias]]
        size = facts.alias_sizes[alias]
        for interval in charged:
            start = interval.start + 1
            end = interval.end + 1
            difference[start] += size
            difference[end + 1] -= size

    pressure: dict[str, tuple[int, ...]] = {}
    for device_id, difference in differences.items():
        values: list[int] = []
        current = 0
        for index in range(boundary_count):
            current += difference[index]
            values.append(current)
        pressure[device_id] = tuple(values)

    mutable = {device_id: list(values) for device_id, values in pressure.items()}
    for task, reservations in enumerate(facts.output_reservations):
        boundary = task - 1
        for alias in reservations:
            if any(span.contains(boundary) for span in charged_spans[alias]):
                continue
            mutable[facts.alias_devices[alias]][task] += facts.alias_sizes[alias]
    return {device_id: tuple(values) for device_id, values in mutable.items()}


def _update_pressure_for_alias(
    facts: PlanningFacts,
    previous: ResidencyPlan,
    current: ResidencyPlan,
    alias: int,
    pressure: dict[str, list[int]],
    *,
    prefetch_headroom: bool,
) -> None:
    """Update a pressure sweep after one alias's residency spans change."""

    def charged(plan: ResidencyPlan) -> tuple[Span, ...]:
        values: list[Span] = []
        for span in plan.spans[alias]:
            interval = _charged_span(
                facts,
                alias,
                span,
                prefetch_headroom=prefetch_headroom,
            )
            if interval is not None:
                values.append(interval)
        return tuple(values)

    previous_spans = charged(previous)
    current_spans = charged(current)
    device_pressure = pressure[facts.alias_devices[alias]]
    size = facts.alias_sizes[alias]
    for boundary in range(-1, facts.last_boundary + 1):
        task = boundary + 1
        reserved = (
            0 <= task < len(facts.output_reservations)
            and alias in facts.output_reservations[task]
        )
        previous_contributes = reserved or any(
            span.contains(boundary) for span in previous_spans
        )
        current_contributes = reserved or any(
            span.contains(boundary) for span in current_spans
        )
        device_pressure[boundary + 1] += size * (
            int(current_contributes) - int(previous_contributes)
        )


def seed_residency(
    facts: PlanningFacts,
    config: SimulationConfig,
    placement: InitialPlacement,
    *,
    initial_capacity_by_device: dict[str, int] | None = None,
) -> ResidencyPlan:
    """Build the anchor hull and optionally preplace fitting host objects."""

    seed = _seed_from_anchors(facts.anchors)
    if placement is InitialPlacement.REQUIRED:
        return seed

    cold_aliases = tuple(
        alias
        for alias, location in enumerate(facts.initial_locations)
        if location is MemoryLocation.HOST
        and facts.access_events[alias]
        and min(task for _boundary, task in facts.access_events[alias]) > 0
    )
    device_config = {item.device_id: item for item in config.devices}

    def transfer_time(alias: int) -> int:
        device = device_config[facts.alias_devices[alias]]
        return (
            device.h2d_latency_ns
            + (
                facts.alias_sizes[alias] * 1_000_000_000
                + device.h2d_bandwidth_bytes_per_second
                - 1
            )
            // device.h2d_bandwidth_bytes_per_second
        )

    first_use = {
        alias: min(task for _boundary, task in facts.access_events[alias])
        for alias in cold_aliases
    }
    deadline = {
        alias: (
            0
            if first_use[alias] == 0
            else facts.task_ideal_end_ns[first_use[alias] - 1]
        )
        for alias in cold_aliases
    }
    miss = {alias: 0 for alias in cold_aliases}
    cursor_by_device = {
        device_id: (facts.task_ideal_end_ns[0] if facts.tasks else 0)
        for device_id in facts.object_capacity_by_device
    }
    for alias in sorted(
        cold_aliases,
        key=lambda value: (deadline[value], first_use[value], value),
    ):
        device_id = facts.alias_devices[alias]
        finish = cursor_by_device[device_id] + transfer_time(alias)
        miss[alias] = max(finish - deadline[alias], 0)
        cursor_by_device[device_id] = finish
    first_task_end = facts.task_ideal_end_ns[0] if facts.tasks else 0
    candidates = sorted(
        cold_aliases,
        key=lambda alias: (
            first_use[alias],
            max(deadline[alias] - first_task_end - transfer_time(alias), 0),
            -miss[alias],
            -facts.alias_sizes[alias],
            alias,
        ),
    )
    initial_bytes = {
        device_id: boundary_bytes(facts, seed, -1, device_id)
        for device_id in facts.object_capacity_by_device
    }
    initial_capacity = initial_capacity_by_device or facts.object_capacity_by_device
    anchors = list(facts.anchors)
    spans = list(seed.spans)
    for alias in candidates:
        device_id = facts.alias_devices[alias]
        proposed_bytes = initial_bytes[device_id] + facts.alias_sizes[alias]
        if proposed_bytes > initial_capacity[device_id]:
            continue
        anchors[alias] = frozenset((*anchors[alias], -1))
        spans[alias] = (Span(min(anchors[alias]), max(anchors[alias])),)
        initial_bytes[device_id] = proposed_bytes
    return ResidencyPlan(tuple(spans), tuple(anchors))


def _gap_containing(
    anchors: frozenset[int], span: Span, boundary: int
) -> tuple[int, int] | None:
    if boundary in anchors or not span.contains(boundary):
        return None
    start = boundary
    while start > span.start and start - 1 not in anchors:
        start -= 1
    end = boundary
    while end < span.end and end + 1 not in anchors:
        end += 1
    return start, end


def legal_cuts(
    facts: PlanningFacts,
    plan: ResidencyPlan,
    boundary: int,
    device_id: str,
) -> tuple[Cut, ...]:
    cuts: list[Cut] = []
    for alias, spans in enumerate(plan.spans):
        if facts.alias_devices[alias] != device_id:
            continue
        for span_index, span in enumerate(spans):
            if (
                span.start == -1
                and facts.initial_locations[alias] is MemoryLocation.HOST
                and -1 not in facts.anchors[alias]
            ):
                required = sorted(
                    anchor for anchor in facts.anchors[alias] if anchor > -1
                )
                if required and boundary < required[0]:
                    # Greedy preplacement is provisional.  Removing this
                    # prefix means "leave it in host memory initially" and
                    # needs no device-side departure action.
                    cuts.append(Cut(alias, span_index, -1, required[0] - 1))
                    continue
            gap = _gap_containing(plan.anchors[alias], span, boundary)
            if gap is None:
                can_split_after = (
                    span.contains(boundary)
                    and boundary in plan.anchors[alias]
                    and boundary < span.end
                    and not any(
                        anchor == boundary and task > boundary
                        for anchor, task in facts.access_events[alias]
                    )
                )
                if not can_split_after:
                    continue
                start, end = boundary + 1, boundary
            else:
                start, end = gap
            # A removed initial boundary has no task at which its departure can
            # be submitted. Initial placement is therefore an anchor, not a cut.
            if start <= -1:
                continue
            cuts.append(Cut(alias, span_index, start, end))
    return tuple(cuts)


def _transfer_runtime_ns(
    facts: PlanningFacts,
    config: SimulationConfig,
    alias: int,
    *,
    to_device: bool,
) -> int:
    device_id = facts.alias_devices[alias]
    device = next(item for item in config.devices if item.device_id == device_id)
    bandwidth = (
        device.h2d_bandwidth_bytes_per_second
        if to_device
        else device.d2h_bandwidth_bytes_per_second
    )
    latency = device.h2d_latency_ns if to_device else device.d2h_latency_ns
    size = facts.alias_sizes[alias]
    return latency + (size * 1_000_000_000 + bandwidth - 1) // bandwidth


def _writeback_required(facts: PlanningFacts, cut: Cut) -> bool:
    if not facts.alias_retain_host[cut.alias]:
        return True
    departure = cut.start - 1
    return any(boundary <= departure for boundary in facts.write_boundaries[cut.alias])


def _cut_score(
    facts: PlanningFacts,
    config: SimulationConfig,
    cut: Cut,
    score_kind: str,
) -> tuple[int, ...]:
    departure = cut.start - 1
    entry = cut.end + 1
    writeback = int(_writeback_required(facts, cut))
    h2d_ns = _transfer_runtime_ns(facts, config, cut.alias, to_device=True)
    d2h_ns = (
        _transfer_runtime_ns(facts, config, cut.alias, to_device=False)
        if writeback
        else 0
    )
    departure_time = facts.task_ideal_end_ns[departure] if departure >= 0 else 0
    deadline_task = min(entry + 1, len(facts.tasks) - 1)
    deadline = facts.task_ideal_end_ns[deadline_task - 1] if deadline_task > 0 else 0
    exposed = max(departure_time + d2h_ns + h2d_ns - deadline, 0)
    first_use_task = facts.first_input_tasks[cut.alias]
    common = (
        writeback,
        -int(cut.start <= -1),
        -first_use_task,
        -facts.alias_sizes[cut.alias],
        -cut.length,
        cut.alias,
        cut.start,
    )
    if score_kind == "min-stall":
        return (exposed, *common)
    return common


def apply_cut(plan: ResidencyPlan, cut: Cut) -> ResidencyPlan:
    spans = list(plan.spans)
    alias_spans = list(spans[cut.alias])
    original = alias_spans.pop(cut.span_index)
    replacements: list[Span] = []
    if original.start < cut.start:
        replacements.append(Span(original.start, cut.start - 1))
    if cut.end < original.end:
        replacements.append(Span(cut.end + 1, original.end))
    alias_spans.extend(replacements)
    alias_spans.sort()
    spans[cut.alias] = tuple(alias_spans)
    return ResidencyPlan(tuple(spans), plan.anchors)


def strategy_object_capacity(
    facts: PlanningFacts,
    device_id: str,
    strategy: str,
) -> int:
    del strategy
    return facts.object_capacity_by_device[device_id]


def reduce_pressure(
    facts: PlanningFacts,
    config: SimulationConfig,
    seed: ResidencyPlan,
    strategy: str,
    *,
    extra_pressure: dict[tuple[str, int], int] | None = None,
    score_cache: dict[tuple[Cut, str], tuple[int, ...]] | None = None,
) -> ResidencyPlan:
    """Greedily remove anchor-free residency until analytic capacity fits."""

    plan = seed
    additions = extra_pressure or {}
    score_kind = "min-transfer" if strategy.endswith("transfer") else "min-stall"
    prefetch_headroom = strategy.startswith("headroom")
    pressure = {
        device_id: list(values)
        for device_id, values in _pressure_by_device(
            facts,
            plan,
            prefetch_headroom=prefetch_headroom,
        ).items()
    }
    while True:
        selected: tuple[str, int, int, int] | None = None
        for boundary in range(-1, facts.last_boundary + 1):
            for device_id in facts.object_capacity_by_device:
                used = pressure[device_id][boundary + 1]
                used += additions.get((device_id, boundary), 0)
                capacity = strategy_object_capacity(facts, device_id, strategy)
                if used <= capacity:
                    continue
                excess = used - capacity
                candidate = (device_id, boundary, excess, used)
                if selected is None or (-excess, boundary, device_id) < (
                    -selected[2],
                    selected[1],
                    selected[0],
                ):
                    selected = candidate
        if selected is None:
            return plan
        device_id, boundary, _excess, used = selected
        cuts = legal_cuts(facts, plan, boundary, device_id)
        if not cuts:
            task_index = boundary + 1
            task_id = (
                facts.tasks[task_index].task_id
                if 0 <= task_index < len(facts.tasks)
                else None
            )
            raise PressureFitInfeasibleError(
                f"no legal residency cut can relieve {used} bytes at "
                f"boundary {boundary} on {device_id!r}; capacity is "
                f"{facts.object_capacity_by_device[device_id]}",
                kind="analytic_capacity",
                device_id=device_id,
                boundary_task_id=task_id,
                required_bytes=used,
                capacity_bytes=facts.object_capacity_by_device[device_id],
            )

        def score(cut: Cut) -> tuple[int, ...]:
            if score_cache is None:
                return _cut_score(facts, config, cut, score_kind)
            key = (cut, score_kind)
            cached = score_cache.get(key)
            if cached is not None:
                return cached
            value = _cut_score(facts, config, cut, score_kind)
            score_cache[key] = value
            return value

        chosen = min(cuts, key=score)
        previous = plan
        plan = apply_cut(plan, chosen)
        _update_pressure_for_alias(
            facts,
            previous,
            plan,
            chosen.alias,
            pressure,
            prefetch_headroom=prefetch_headroom,
        )


def extend_interval_entries(
    facts: PlanningFacts,
    plan: ResidencyPlan,
) -> ResidencyPlan:
    """Move later entries earlier only while exact analytic pressure fits."""

    current = plan
    pressure = {
        device_id: list(values)
        for device_id, values in _pressure_by_device(facts, current).items()
    }
    for alias in range(len(facts.alias_ids)):
        span_index = 1
        while span_index < len(current.spans[alias]):
            span = current.spans[alias][span_index]
            previous = current.spans[alias][span_index - 1]
            candidate_start = span.start - 1
            if candidate_start <= previous.end:
                span_index += 1
                continue
            spans = list(current.spans)
            alias_spans = list(spans[alias])
            alias_spans[span_index] = Span(candidate_start, span.end)
            spans[alias] = tuple(alias_spans)
            proposed = ResidencyPlan(tuple(spans), current.anchors)
            device_id = facts.alias_devices[alias]
            task = candidate_start + 1
            reserved = (
                0 <= task < len(facts.output_reservations)
                and alias in facts.output_reservations[task]
            )
            added = 0 if reserved else facts.alias_sizes[alias]
            if (
                pressure[device_id][candidate_start + 1] + added
                <= facts.object_capacity_by_device[device_id]
            ):
                current = proposed
                pressure[device_id][candidate_start + 1] += added
                continue
            span_index += 1
    return current


def assert_required_floor(facts: PlanningFacts) -> None:
    """Fail early when anchors plus fresh outputs exceed object capacity."""

    minimal = ResidencyPlan(
        tuple(
            tuple(Span(value, value) for value in sorted(anchors))
            for anchors in facts.anchors
        ),
        facts.anchors,
    )
    pressure = _pressure_by_device(facts, minimal)
    for boundary in range(-1, facts.last_boundary + 1):
        for device_id, capacity in facts.object_capacity_by_device.items():
            required = pressure[device_id][boundary + 1]
            if required > capacity:
                task_index = boundary + 1
                task_id = (
                    facts.tasks[task_index].task_id
                    if 0 <= task_index < len(facts.tasks)
                    else None
                )
                raise PressureFitInfeasibleError(
                    f"required inputs and outputs need {required} bytes at "
                    f"{task_id or 'initialization'} on {device_id!r}, exceeding "
                    f"object capacity {capacity}",
                    kind="required_capacity",
                    device_id=device_id,
                    boundary_task_id=task_id,
                    required_bytes=required,
                    capacity_bytes=capacity,
                )


__all__ = [
    "ResidencyPlan",
    "Span",
    "assert_required_floor",
    "boundary_bytes",
    "extend_interval_entries",
    "pressure_by_boundary",
    "reduce_pressure",
    "seed_residency",
    "strategy_object_capacity",
]
