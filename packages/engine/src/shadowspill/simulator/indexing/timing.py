"""Project measured timing into the C simulator's immutable tables."""

from __future__ import annotations

import bisect
import ctypes
import math

from shadowspill.ir import TaskOverlapCost, TaskProfile, TransferPhase

from ..capi import NO_INDEX, CTaskResourcePhase
from .arrays import _Arena


def interpolate_cost(
    costs: tuple[TaskOverlapCost, ...], timing_class: str, size: int
) -> tuple[float, float]:
    samples = sorted(
        (item for item in costs if item.timing_class == timing_class),
        key=lambda item: item.bytes,
    )
    if not samples:
        return 1.0, 1.0
    index = bisect.bisect_left([item.bytes for item in samples], size)
    if index == 0 or index == len(samples):
        item = samples[0] if index == 0 else samples[-1]
        return item.task_rate, item.transfer_rate
    lower, upper = samples[index - 1 : index + 1]
    # log1p keeps zero-byte latency samples defined and distinct from one byte.
    fraction = (math.log1p(size) - math.log1p(lower.bytes)) / (
        math.log1p(upper.bytes) - math.log1p(lower.bytes)
    )
    return (
        lower.task_rate + fraction * (upper.task_rate - lower.task_rate),
        lower.transfer_rate + fraction * (upper.transfer_rate - lower.transfer_rate),
    )


def timing_tables(
    profiles: tuple[TaskProfile, ...],
    transfers: tuple[TransferPhase, ...],
    resources: dict[str, int],
    arena: _Arena,
) -> dict[str, object]:
    offsets = [0]
    phases = []
    for profile in profiles:
        for phase in profile.resource_phases:
            resource = (
                NO_INDEX
                if phase.resource is None
                else resources.setdefault(phase.resource, len(resources))
            )
            phases.append(CTaskResourcePhase(phase.duration_ns, resource))
        offsets.append(len(phases))
    classes = {item.timing_class for p in profiles for item in p.overlap_costs}
    categories: dict[tuple[str, int], int] = {}
    phase_categories = []
    for transfer in transfers:
        if transfer.timing_class is None or transfer.timing_class not in classes:
            phase_categories.append(NO_INDEX)
        else:
            key = (transfer.timing_class, transfer.bytes)
            phase_categories.append(categories.setdefault(key, len(categories)))
    # Structural profiles repeat thousands of times. Keep one short C
    # rate row per distinct profile; candidate validation need not scan a dense
    # task-by-transfer matrix for every schedule the planner evaluates.
    profile_indices: dict[str, int] = {}
    task_profiles = []
    task_rates: list[float] = []
    transfer_rates: list[float] = []
    for profile in profiles:
        index = profile_indices.get(profile.profile_id)
        if index is None:
            index = len(profile_indices)
            profile_indices[profile.profile_id] = index
            row = tuple(
                interpolate_cost(profile.overlap_costs, timing_class, size)
                for timing_class, size in categories
            )
            task_rates.extend(item[0] for item in row)
            transfer_rates.extend(item[1] for item in row)
        task_profiles.append(index)

    def doubles(values: list[float]) -> object:
        return arena.keep((ctypes.c_double * len(values))(*values)) if values else None

    return {
        "task_phase_count": len(phases),
        "task_phase_offsets": arena.u32(tuple(offsets)) if phases else None,
        "task_phases": arena.keep((CTaskResourcePhase * len(phases))(*phases))
        if phases
        else None,
        "overlap_category_count": len(categories),
        "overlap_profile_count": len(profile_indices) if categories else 0,
        "task_overlap_profiles": arena.u32(tuple(task_profiles))
        if categories
        else None,
        "transfer_overlap_categories": arena.u32(tuple(phase_categories))
        if categories
        else None,
        "task_overlap_rates": doubles(task_rates),
        "transfer_overlap_rates": doubles(transfer_rates),
    }
