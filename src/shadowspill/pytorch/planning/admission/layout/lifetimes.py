"""Construct physical lease lifetimes from one selected schedule."""

from __future__ import annotations

from collections import defaultdict

from shadowspill.ir import MemoryActionKind, MemorySchedule
from shadowspill.runtime import AdmissionReplayOperationKind
from shadowspill.simulator import SimulationResult, TaskInterval, TransferInterval

from ..admission_replay import (
    AdmissionReplayPurpose,
    AdmissionReplayStep,
    _LeaseProvenance,
)
from .model import LeaseLifetime


def build_lease_lifetimes(
    operations: tuple[AdmissionReplayStep, ...],
    lease_provenance: dict[int, _LeaseProvenance],
    schedule: MemorySchedule,
    simulation: SimulationResult,
) -> tuple[LeaseLifetime, ...]:
    """Return conservative per-lease lifetimes for physical placement."""

    task_intervals = {item.task_id: item for item in simulation.task_intervals}
    transfer_intervals = _transfer_intervals_by_action(schedule, simulation)
    starts: dict[int, tuple[int, int, int]] = {}
    retirements: dict[int, AdmissionReplayStep] = {}
    for index, step in enumerate(operations):
        operation = step.operation
        if operation.kind in {
            AdmissionReplayOperationKind.ACQUIRE,
            AdmissionReplayOperationKind.RESERVE,
        }:
            starts.setdefault(
                operation.lease_id,
                (index, operation.bytes, operation.alignment),
            )
        elif operation.kind in {
            AdmissionReplayOperationKind.BEGIN_RETIREMENT,
            AdmissionReplayOperationKind.RELEASE,
        }:
            retirements.setdefault(operation.lease_id, step)

    terminal_time = simulation.makespan_ns + 1
    terminal_boundary = len(operations) + 1
    result: list[LeaseLifetime] = []
    for lease_id, (causal_start, bytes_, alignment) in sorted(starts.items()):
        provenance = lease_provenance[lease_id]
        predicted_start = _predicted_start(
            provenance.purpose,
            provenance.task_id,
            task_intervals,
        )
        retirement = retirements.get(lease_id)
        if retirement is None:
            predicted_end = terminal_time
            causal_end = terminal_boundary
        else:
            predicted_end = _predicted_end(
                retirement,
                task_intervals,
                transfer_intervals,
            )
            causal_end = retirement.operation.sequence
        if predicted_end < predicted_start:
            raise ValueError(
                f"lease {lease_id} ends at {predicted_end} before its "
                f"start at {predicted_start}"
            )
        result.append(
            LeaseLifetime(
                lease_id=lease_id,
                bytes=bytes_,
                alignment=alignment,
                predicted_start_ns=predicted_start,
                predicted_end_ns=predicted_end,
                causal_start=causal_start,
                causal_end=causal_end,
                purpose=provenance.purpose,
                task_id=provenance.task_id,
                alias_group_id=provenance.alias_group_id,
                action_index=provenance.action_index,
            )
        )
    return tuple(result)


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


def _predicted_end(
    retirement: AdmissionReplayStep,
    task_intervals: dict[str, TaskInterval],
    transfer_intervals: dict[int, TransferInterval],
) -> int:
    if retirement.purpose is AdmissionReplayPurpose.EVICTION:
        if retirement.action_index is None:
            raise ValueError("eviction retirement lacks an action identity")
        return transfer_intervals[retirement.action_index].end_ns
    if retirement.task_id is None:
        raise ValueError(f"{retirement.purpose.value} retirement lacks a task identity")
    return int(task_intervals[retirement.task_id].end_ns)


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


__all__ = ["build_lease_lifetimes"]
