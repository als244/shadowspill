"""Record the exact state displayed by a residency-cut frame."""

from __future__ import annotations

import numpy as np

from ...data import short_task
from ...pressure import ROLE_GROUPS
from .geometry import recompute_endpoint


def record_frame(
    self,
    view,
    rid,
    count,
    phase,
    event,
    batch_start,
    pressure,
    capacity,
    reduction_context,
    composition,
    displayed_pressure,
    ceiling,
    current_task,
    current_memory,
    zoom_progress,
):
    d = self.data
    row = view.row
    selected = view.selected
    next_use = view.next_use
    departure = view.departure
    highlight_cut = view.highlight_cut
    journey = view.journey
    applied = view.applied
    removed = view.removed
    position = view.position
    lo = view.lo
    hi = view.hi
    xmin = view.xmin
    xmax = view.xmax
    prior_use = view.prior_use
    focus = view.focus
    local = view.local
    self.last_record = {
        "axis": d.axis,
        "reduction": rid,
        "reduction_context": reduction_context,
        "cut_count": count,
        "phase": phase,
        "event_index": event,
        "batch_start": batch_start,
        "boundary_position": position,
        "current_task": (
            {
                "task_position": position,
                "task_id": current_task["task_id"],
                "label": short_task(current_task),
                "required_memory": current_memory,
            }
            if current_task
            else None
        ),
        "focus_aliases": [int(a) for a in focus],
        "all_local_aliases": local,
        "focus_task_range": [lo, hi],
        "focus_axis_limits": [float(xmin), float(xmax)],
        "recompute_prefixes": [
            {
                "task_position": i,
                "task_id": d.tasks[i]["task_id"],
                "label": short_task(d.tasks[i], phase="recompute"),
                "duration_ns": d.tasks[i]["recompute_prefix_ns"],
                "save_backward_duration_ns": d.tasks[i]["save_backward_duration_ns"],
                "axis_start": float(d.edges[i]),
                "axis_end": recompute_endpoint(d, i),
            }
            for i in range(lo, hi)
            if d.tasks[i].get("recompute_prefix_ns", 0) > 0
        ],
        "peak_pressure_bytes": int(pressure.max()),
        "display_peak_pressure_bytes": int(displayed_pressure.max()),
        "display_budget_bytes": self.composition.flat_budget,
        "resident_pool_bytes": self.pressure_offset,
        "pressure_ylim_gib": [0, ceiling],
        "pressure_roles_at_boundary": {
            name: int(composition[i, position])
            for i, (name, _, _) in enumerate(ROLE_GROUPS)
        },
        "max_excess_bytes": max(
            0, int(np.max(pressure.astype(np.int64) - capacity.astype(np.int64)))
        ),
    }
    if journey:
        self.last_record["object_journey"] = {
            "alias_index": selected,
            "prior_use_position": prior_use,
            "departure_position": departure,
            "next_use_position": next_use,
            "intervening_tasks": next_use - departure,
            "zoom_progress": zoom_progress,
        }
    if highlight_cut:
        self.last_record["cut_highlight"] = {
            "alias_index": selected,
            "status": "applied" if applied else "selected",
            "removed_intervals": removed,
            "boundary_only": not removed,
            "action": "evict" if int(row[8]) else "release",
            "departure_position": departure,
            "next_use_position": next_use,
            "compute_gap_ns": (
                d.gap_duration_ns(row, next_use) if next_use is not None else None
            ),
            "compute_until_next_use_ns": (
                d.time_to_use_ns(row, next_use) if next_use is not None else None
            ),
        }
