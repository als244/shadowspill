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


def _with_anchor(
    anchors: tuple[frozenset[int], ...], alias: int, boundary: int
) -> tuple[frozenset[int], ...]:
    changed = list(anchors)
    changed[alias] = frozenset((*changed[alias], boundary))
    return tuple(changed)


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
    return tuple(
        {
            device_id: boundary_bytes(facts, plan, boundary, device_id)
            for device_id in facts.object_capacity_by_device
        }
        for boundary in range(-1, facts.last_boundary + 1)
    )


def seed_residency(
    facts: PlanningFacts,
    config: SimulationConfig,
    placement: InitialPlacement,
    *,
    initial_capacity_by_device: dict[str, int] | None = None,
) -> ResidencyPlan:
    """Build the anchor hull and optionally preplace fitting host objects."""

    anchors = facts.anchors
    seed = _seed_from_anchors(anchors)
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
    for alias in candidates:
        device_id = facts.alias_devices[alias]
        proposed_bytes = initial_bytes[device_id] + facts.alias_sizes[alias]
        if proposed_bytes > initial_capacity[device_id]:
            continue
        anchors = _with_anchor(anchors, alias, -1)
        seed = _seed_from_anchors(anchors)
        initial_bytes[device_id] = proposed_bytes
    return seed


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
    first_use_task = min(
        (
            task
            for boundary, task in facts.access_events[cut.alias]
            if boundary == task - 1
        ),
        default=len(facts.tasks),
    )
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
    spans = [list(values) for values in plan.spans]
    original = spans[cut.alias].pop(cut.span_index)
    replacements: list[Span] = []
    if original.start < cut.start:
        replacements.append(Span(original.start, cut.start - 1))
    if cut.end < original.end:
        replacements.append(Span(cut.end + 1, original.end))
    spans[cut.alias].extend(replacements)
    spans[cut.alias].sort()
    return ResidencyPlan(tuple(tuple(values) for values in spans), plan.anchors)


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
) -> ResidencyPlan:
    """Greedily remove anchor-free residency until analytic capacity fits."""

    plan = seed
    additions = extra_pressure or {}
    score_kind = "min-transfer" if strategy.endswith("transfer") else "min-stall"
    while True:
        selected: tuple[str, int, int, int] | None = None
        for boundary in range(-1, facts.last_boundary + 1):
            for device_id in facts.object_capacity_by_device:
                used = boundary_bytes(
                    facts,
                    plan,
                    boundary,
                    device_id,
                    prefetch_headroom=strategy.startswith("headroom"),
                )
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
        chosen = min(
            cuts,
            key=lambda cut: _cut_score(facts, config, cut, score_kind),
        )
        plan = apply_cut(plan, chosen)


def extend_interval_entries(
    facts: PlanningFacts,
    plan: ResidencyPlan,
) -> ResidencyPlan:
    """Move later entries earlier only while exact analytic pressure fits."""

    current = plan
    for alias in range(len(facts.alias_ids)):
        span_index = 1
        while span_index < len(current.spans[alias]):
            span = current.spans[alias][span_index]
            previous = current.spans[alias][span_index - 1]
            candidate_start = span.start - 1
            if candidate_start <= previous.end:
                span_index += 1
                continue
            spans = [list(values) for values in current.spans]
            spans[alias][span_index] = Span(candidate_start, span.end)
            proposed = ResidencyPlan(
                tuple(tuple(values) for values in spans), current.anchors
            )
            device_id = facts.alias_devices[alias]
            if (
                boundary_bytes(facts, proposed, candidate_start, device_id)
                <= (facts.object_capacity_by_device[device_id])
            ):
                current = proposed
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
    for boundary in range(-1, facts.last_boundary + 1):
        for device_id, capacity in facts.object_capacity_by_device.items():
            required = boundary_bytes(facts, minimal, boundary, device_id)
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
