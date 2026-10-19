"""Conservative shared placement for rank-local temporary allocation sizes.

Kernel autotuners may pick different tile sizes on otherwise identical ranks.
Only anonymous allocation sizes may differ here. Ownership, allocation/reuse
order, capacities and the peak workspace contract must still agree. Original
measurements remain the runtime's exact allocation contracts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any, cast

from shadowspill.planner.admission import AdmissionFacts, TaskAllocationStepKind

from ._control import Control


def reconcile(facts: AdmissionFacts, control: Control) -> AdmissionFacts | None:
    shape = cast(dict[str, Any], facts.to_dict())
    shape["device_id"] = "local"
    vectors: list[tuple[int, ...]] = []
    indices = []
    unique = {}
    for task in shape["tasks"]:
        sizes = []
        for step in task["allocation_steps"]:
            if step["kind"] == "allocate" and step["output_alias_group_id"] is None:
                sizes.append(step["charged_bytes"])
                step["charged_bytes"] = 0
        vector = tuple(sizes)
        if vector not in unique:
            unique[vector] = len(vectors)
            vectors.append(vector)
        indices.append(unique[vector])
    digest = hashlib.sha256(
        json.dumps(shape, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    signatures = control.exchange("symmetry/allocation_structure", digest)
    if any(value != signatures[0] for value in signatures):
        return None
    peers = control.exchange(
        "symmetry/anonymous_allocation_sizes", {"vectors": vectors, "indices": indices}
    )
    tasks = []
    for index, task in enumerate(facts.tasks):
        maxima = iter(
            max(values)
            for values in zip(
                *(peer["vectors"][peer["indices"][index]] for peer in peers),
                strict=True,
            )
        )
        steps = tuple(
            replace(step, charged_bytes=next(maxima))
            if step.kind is TaskAllocationStepKind.ALLOCATE
            and step.output_alias_group_id is None
            else step
            for step in task.allocation_steps
        )
        # Combining per-allocation maxima can create a new peak even when all
        # original peaks match. Refuse that case instead of silently changing
        # the already verified logical workspace/capacity contract.
        live: dict[int, int] = {}
        peak: tuple[int, ...] = ()
        peak_bytes = 0
        for step in steps:
            if step.kind is TaskAllocationStepKind.ALLOCATE:
                if step.output_alias_group_id is None:
                    live[step.allocation_ordinal] = step.charged_bytes
            else:
                live.pop(step.allocation_ordinal, None)
            current = sum(live.values())
            if current > peak_bytes:
                peak_bytes, peak = current, tuple(sorted(live.values()))
        if peak != task.workspace_extents:
            return None
        replacements = set(task.replacement_aliases)
        if replacements:
            peaks = []
            for trace in (task.allocation_steps, steps):
                live = {}
                maximum = 0
                for step in trace:
                    if step.kind is TaskAllocationStepKind.ALLOCATE:
                        if (
                            step.output_alias_group_id is None
                            or step.output_alias_group_id in replacements
                        ):
                            live[step.allocation_ordinal] = step.charged_bytes
                    else:
                        live.pop(step.allocation_ordinal, None)
                    maximum = max(maximum, sum(live.values()))
                peaks.append(maximum)
            if peaks[0] != peaks[1]:
                return None
        tasks.append(replace(task, allocation_steps=steps))
    return replace(facts, tasks=tuple(tasks))
