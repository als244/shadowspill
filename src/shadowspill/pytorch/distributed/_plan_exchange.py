"""Exchange verified planning problems and portable schedules between ranks.

No compiled code, runtime tensors, or allocator bindings travel here. Received
schedules are reconstructed and admitted against each receiving rank's facts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any

from shadowspill.planner.admission.refinement import FixedLayoutSelection
from shadowspill.planner.annotated_plan import AnnotatedProgramPlan
from shadowspill.planner.plan_store import certified_result
from shadowspill.planner.serialization import _simulation_config_to_dict
from shadowspill.planner.serialization.simulation_result import simulation_result_value
from shadowspill.store import ArtifactStore, atomic_text


def pack(
    record: FixedLayoutSelection | AnnotatedProgramPlan, *, compact_facts: bool = False
) -> dict[str, Any]:
    """Portable schedules; verified peers can reference their local facts.

    Persisted records remain self-contained. On the wire, the capacity and a
    device-independent digest identify the already verified allocation facts,
    avoiding a second copy of every task's allocation steps on every rank.
    """
    result = (
        certified_result(record.result, record.admission)
        if isinstance(record, FixedLayoutSelection)
        else record.result
    )
    facts = (
        record.facts
        if isinstance(record, FixedLayoutSelection)
        else record.effective_facts
    )
    layout = (
        record.admission.layout
        if isinstance(record, FixedLayoutSelection)
        else record.fixed_layout
    )
    return {
        "schedule": result.schedule.to_dict(),
        "choices": [item.to_dict() for item in result.selections],
        "config": _simulation_config_to_dict(result.simulation_config),
        "options": result.search_options.to_dict(),
        "simulation": simulation_result_value(result.simulation),
        "diagnostics": result.diagnostics.to_dict(),
        "resident_slice": result.resident_slice.to_dict(),
        "facts": (
            {
                "reference": replace(facts, device_id="local").digest,
                "object_capacity_bytes": facts.object_capacity_bytes,
            }
            if compact_facts
            else facts.to_dict()
        ),
        "scratch": layout.scratch_reserve_bytes,
        "resolutions": [
            {
                "selection_id": row.selection_id,
                "candidate_id": row.candidate_id,
                "schedule": row.schedule.to_dict(),
                "simulation": simulation_result_value(row.simulation),
                "choices": [item.to_dict() for item in row.selections],
                "resident_slice": row.resident_slice.to_dict(),
            }
            for row in result.resolutions
        ],
    }


def record_plan(
    store: ArtifactStore, record: FixedLayoutSelection | AnnotatedProgramPlan
) -> str | None:
    """Keep received schedules, including every retained resolution, per rank."""
    if not store.plan_policy.write_enabled:
        return None
    program = record.result.program
    store.archive_program(program)
    payload = {"version": 1, "program": program.digest, "plan": pack(record)}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    identity = hashlib.sha256(encoded.encode()).hexdigest()
    path = store.planning / "distributed" / "plans" / f"{identity}.json"
    atomic_text(path, encoded)
    store.record(
        category="planning",
        kind="distributed_plan",
        digest=identity,
        path=path,
        access="write",
        schema="distributed_plan/v1",
        dependencies=(program.digest,),
    )
    return str(path)
