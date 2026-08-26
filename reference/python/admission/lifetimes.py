"""Readable lease-lifetime construction; production uses the planner.

This is the algorithm `shadowspill_build_lease_lifetimes` implements, written
for reading rather than speed: it is the oracle the compiled pass is
differentially tested against, and the baseline its speedup is measured
against.

Turn one schedule's operations into the lease lifetimes a layout places.

The compiled walk reports operations as indexed columns and, per lease, which
operation creates it and which retires it. That second part is what makes this
a pass over leases rather than over operations: a lease's lifetime is decided
by exactly those two, and most operations decide nothing.

Alongside the lifetimes it produces the lease maps the certificate needs,
because they fall out of the same pass. The rules it follows - where an
operation sits, why a lease exists, and the transitions that emit no operation
at all - are specified in docs/architecture/admission-leases.md.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from shadowspill.ir import MemoryActionKind, MemorySchedule
from shadowspill.planner.admission.admission_replay import (
    AdmissionReplayPurpose,
)
from shadowspill.planner.admission.layout.model import LeaseLifetime
from shadowspill.planner.admission.operations import AdmissionOperations
from shadowspill.planner.admission.setup import AdmissionSetup
from shadowspill.simulator import SimulationResult, TaskInterval, TransferInterval

#: Compiled purpose codes, in the order `ShadowSpillAdmissionPurpose` declares.
_PURPOSES = (
    AdmissionReplayPurpose.INITIAL_OBJECT,
    AdmissionReplayPurpose.TASK_WORKSPACE,
    AdmissionReplayPurpose.TASK_OUTPUT,
    AdmissionReplayPurpose.MUTATION_REPLACEMENT,
    AdmissionReplayPurpose.RELEASE,
    AdmissionReplayPurpose.EVICTION,
    AdmissionReplayPurpose.FETCH_DESTINATION,
    AdmissionReplayPurpose.TERMINAL_COMPLETION,
)

#: Initial residency names neither a task nor an action, and its index carries
#: no meaning; action boundaries name an action and its triggering task.
_INITIAL_BOUNDARY = 0
_ACTION_BOUNDARIES = frozenset({3, 4})


@dataclass(frozen=True, slots=True)
class LeaseLayoutInputs:
    """Everything the placer and the certificate need from one schedule."""

    lifetimes: tuple[LeaseLifetime, ...]
    active_aliases: dict[str, int]
    initial_alias_leases: dict[str, int]
    task_allocation_leases: dict[tuple[str, int], int]
    action_destination_leases: dict[int, int]


def build_lease_layout_inputs(
    operations: AdmissionOperations,
    setup: AdmissionSetup,
    schedule: MemorySchedule,
    simulation: SimulationResult,
) -> LeaseLayoutInputs:
    """Resolve every lease to a lifetime, and collect the certificate maps."""

    task_ids = setup.task_ids
    alias_ids = setup.alias_ids
    allocation_steps = setup.allocation_steps
    trigger_tasks = setup.action_trigger_tasks
    task_intervals = {item.task_id: item for item in simulation.task_intervals}
    transfer_intervals = _transfer_intervals_by_action(schedule, simulation)

    purposes = operations.purposes
    boundaries = operations.boundaries
    indices = operations.indices
    offsets = operations.allocation_offsets
    lease_aliases = operations.lease_aliases
    alias_count = len(lease_aliases)

    # A task allocation's identity comes from its step, not from the operation
    # that acquired the lease: a reallocation gives the same lease a new one
    # and emits no operation of its own.
    slot_of_lease: dict[int, int] = {}
    for lease, start in enumerate(operations.lease_starts):
        offset = offsets[start]
        if offset is not None:
            slot_of_lease[lease] = allocation_steps[offset].slot
    lease_of_slot = {slot: lease for lease, slot in slot_of_lease.items()}
    latest_step = {
        step.slot: step
        for step in allocation_steps
        if step.allocates and step.slot in lease_of_slot
    }

    terminal_time = simulation.makespan_ns + 1
    terminal_boundary = len(operations.kinds) + 1
    lifetimes: list[LeaseLifetime] = []
    active: dict[str, int] = {}
    initial: dict[str, int] = {}
    allocations: dict[tuple[str, int], int] = {}
    destinations: dict[int, int] = {}

    for lease, start in enumerate(operations.lease_starts):
        slot = slot_of_lease.get(lease)
        step = None if slot is None else latest_step.get(slot)
        if step is not None:
            purpose = step.purpose
            task_id: str | None = step.task_id
            alias_id = step.alias_group_id
            action_index: int | None = None
        else:
            purpose = _PURPOSES[purposes[start]]
            boundary = boundaries[start]
            index = indices[start]
            if boundary in _ACTION_BOUNDARIES:
                task_id = task_ids[trigger_tasks[index]]
                action_index = index
            elif boundary == _INITIAL_BOUNDARY:
                task_id = None
                action_index = None
            else:
                task_id = task_ids[index]
                action_index = None
            alias_slot = lease_aliases[lease] if lease < alias_count else None
            alias_id = None if alias_slot is None else alias_ids[alias_slot]

        retire = operations.lease_retires[lease]
        if retire is None:
            predicted_end = terminal_time
            causal_end = terminal_boundary
            if alias_id is not None:
                active[alias_id] = lease
        else:
            predicted_end = _predicted_end(
                _PURPOSES[purposes[retire]],
                boundaries[retire],
                indices[retire],
                task_ids,
                trigger_tasks,
                task_intervals,
                transfer_intervals,
            )
            causal_end = retire

        predicted_start = _predicted_start(purpose, task_id, task_intervals)
        if predicted_end < predicted_start:
            raise ValueError(
                f"lease {lease} ends at {predicted_end} before its "
                f"start at {predicted_start}"
            )
        lifetimes.append(
            LeaseLifetime(
                lease_id=lease,
                bytes=operations.bytes[start],
                alignment=operations.alignments[start],
                predicted_start_ns=predicted_start,
                predicted_end_ns=predicted_end,
                causal_start=start,
                causal_end=causal_end,
                purpose=purpose,
                task_id=task_id,
                alias_group_id=alias_id,
                action_index=action_index,
            )
        )
        if purpose is AdmissionReplayPurpose.INITIAL_OBJECT and alias_id:
            initial[alias_id] = lease
        elif (
            purpose is AdmissionReplayPurpose.FETCH_DESTINATION
            and action_index is not None
        ):
            destinations[action_index] = lease

    for step in allocation_steps:
        if step.allocates:
            lease_id = lease_of_slot.get(step.slot)
            if lease_id is not None:
                allocations[(step.task_id, step.ordinal)] = lease_id

    for source, destination in setup.storage_handoffs:
        moved = active.pop(source, None)
        if moved is not None:
            active[destination] = moved

    return LeaseLayoutInputs(
        lifetimes=tuple(lifetimes),
        active_aliases=active,
        initial_alias_leases=initial,
        task_allocation_leases=allocations,
        action_destination_leases=destinations,
    )


def _predicted_start(
    purpose: AdmissionReplayPurpose,
    task_id: str | None,
    task_intervals: dict[str, TaskInterval],
) -> int:
    if purpose is AdmissionReplayPurpose.INITIAL_OBJECT:
        return 0
    if task_id is None:
        raise ValueError(f"{purpose.value} lease lacks a task identity")
    task = task_intervals[task_id]
    if purpose is AdmissionReplayPurpose.FETCH_DESTINATION:
        return int(task.end_ns)
    return int(task.start_ns)


def _predicted_end(  # type: ignore[no-untyped-def]
    purpose,
    boundary,
    index,
    task_ids,
    trigger_tasks,
    task_intervals,
    transfer_intervals,
) -> int:
    """An eviction frees its address once the copy lands; anything else frees
    it when its task ends."""

    if purpose is AdmissionReplayPurpose.EVICTION:
        if boundary not in _ACTION_BOUNDARIES:
            raise ValueError("eviction retirement lacks an action identity")
        return int(transfer_intervals[index].end_ns)
    if boundary in _ACTION_BOUNDARIES:
        return int(task_intervals[task_ids[trigger_tasks[index]]].end_ns)
    if boundary == _INITIAL_BOUNDARY:
        raise ValueError(f"{purpose.value} retirement lacks a task identity")
    return int(task_intervals[task_ids[index]].end_ns)


def _transfer_intervals_by_action(
    schedule: MemorySchedule,
    simulation: SimulationResult,
) -> dict[int, TransferInterval]:
    by_sequence = {
        (item.direction.value, item.sequence): item
        for item in simulation.transfer_intervals
    }
    sequences: defaultdict[str, int] = defaultdict(int)
    result: dict[int, TransferInterval] = {}
    for index, action in enumerate(schedule.actions):
        if action.kind is MemoryActionKind.RELEASE:
            continue
        direction = "evict" if action.kind is MemoryActionKind.OFFLOAD else "fetch"
        sequence = sequences[direction]
        sequences[direction] += 1
        interval = by_sequence[(direction, sequence)]
        if (
            interval.alias_group_id != action.alias_group_id
            or interval.trigger_task_id != action.trigger_task_id
        ):
            raise ValueError(
                f"simulated {direction} sequence {sequence} does not match "
                f"schedule action {index}"
            )
        result[index] = interval
    return result


__all__ = ["LeaseLayoutInputs", "build_lease_layout_inputs"]
