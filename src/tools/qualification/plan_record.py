"""What a planning call was asked and what it answered, written as evidence.

One record per plan: the framework-free request the search was given, the
answer it returned, and digests over both. A qualification run writes these
beside its results so a plan can be compared against another run without
either run being repeated.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from typing import Any

from shadowspill.planner import ProgramPlanResult
from shadowspill.schema import artifact_schema


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _simulation_value(result: ProgramPlanResult) -> dict[str, Any]:
    return asdict(result.simulation)


def plan_record(
    result: ProgramPlanResult,
    *,
    role: str,
) -> dict[str, Any]:
    """Return the exact framework-free request this search was given, and its answer."""

    request = {
        "program": result.program.to_dict(),
        "initial_residency": [item.to_dict() for item in result.initial_residency],
        "final_residency": [item.to_dict() for item in result.final_residency],
        "simulation_config": asdict(result.simulation_config),
        "search_options": result.search_options.to_dict(),
        "admission": (
            None if result.admission_facts is None else result.admission_facts.to_dict()
        ),
        "placement": (
            None if result.placement_facts is None else result.placement_facts.to_dict()
        ),
    }
    expected = {
        "schedule": result.schedule.to_dict(),
        "selections": [item.to_dict() for item in result.selections],
        "simulation": _simulation_value(result),
        "diagnostics": result.diagnostics.to_dict(),
    }
    stable_expected = dict(expected)
    stable_expected["diagnostics"] = result.diagnostics.stable_dict()
    return {
        "schema": artifact_schema("plan_record"),
        "role": role,
        "request_digest": _digest(request),
        "expected_digest": _digest(stable_expected),
        "program_digest": result.program.digest,
        "schedule_digest": result.schedule.digest,
        "request": request,
        "expected": expected,
    }


def _write_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as output:
            output.write(_canonical(value))
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        with suppress(FileNotFoundError):
            os.unlink(temporary)


def write_plan_records(
    *,
    results: tuple[ProgramPlanResult, ...],
    directory: Path,
) -> list[dict[str, object]]:
    """Persist the initial/recurrent records and return compact artifact evidence."""

    pairs: tuple[tuple[str, ProgramPlanResult], ...]
    if len(results) == 1:
        pairs = (("recurrent", results[0]),)
    elif len(results) == 2:
        pairs = (
            ("initial", results[0]),
            ("recurrent", results[1]),
        )
    else:
        raise ValueError("results do not match initial/recurrent plans")
    evidence: list[dict[str, object]] = []
    for role, result in pairs:
        record = plan_record(result, role=role)
        path = directory / f"{role}.json"
        _write_atomic(path, record)
        evidence.append(
            {
                "role": role,
                "path": str(path),
                "request_digest": record["request_digest"],
                "expected_digest": record["expected_digest"],
                "program_digest": record["program_digest"],
                "schedule_digest": record["schedule_digest"],
            }
        )
    return evidence


__all__ = ["plan_record", "write_plan_records"]
