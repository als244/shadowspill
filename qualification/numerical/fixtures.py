"""Deterministic PressureFit input/output fixtures for planner equivalence."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from typing import Any

from shadowspill.planner import PressureFitResult


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _simulation_value(result: PressureFitResult) -> dict[str, Any]:
    return asdict(result.simulation)


def pressurefit_fixture(
    result: PressureFitResult,
    *,
    role: str,
) -> dict[str, Any]:
    """Return the exact framework-free input and output of `pressurefit()`."""

    request = {
        "program": result.program.to_dict(),
        "initial_residency": [item.to_dict() for item in result.initial_residency],
        "final_residency": [item.to_dict() for item in result.final_residency],
        "simulation_config": asdict(result.simulation_config),
        "options": asdict(result.options),
    }
    expected = {
        "schedule": result.schedule.to_dict(),
        "selections": [item.to_dict() for item in result.selections],
        "simulation": _simulation_value(result),
        "diagnostics": asdict(result.diagnostics),
    }
    return {
        "schema": "shadowspill.pressurefit_fixture/v1",
        "role": role,
        "request_digest": _digest(request),
        "expected_digest": _digest(expected),
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


def write_pressurefit_fixtures(
    *,
    results: tuple[PressureFitResult, ...],
    directory: Path,
) -> list[dict[str, object]]:
    """Persist initial/recurrent fixtures and return compact artifact evidence."""

    pairs: tuple[tuple[str, PressureFitResult], ...]
    if len(results) == 1:
        pairs = (("recurrent", results[0]),)
    elif len(results) == 2:
        pairs = (
            ("initial", results[0]),
            ("recurrent", results[1]),
        )
    else:
        raise ValueError("PressureFit results do not match initial/recurrent plans")
    evidence: list[dict[str, object]] = []
    for role, result in pairs:
        fixture = pressurefit_fixture(result, role=role)
        path = directory / f"{role}.json"
        _write_atomic(path, fixture)
        evidence.append(
            {
                "role": role,
                "path": str(path),
                "request_digest": fixture["request_digest"],
                "expected_digest": fixture["expected_digest"],
                "program_digest": fixture["program_digest"],
                "schedule_digest": fixture["schedule_digest"],
            }
        )
    return evidence


__all__ = ["pressurefit_fixture", "write_pressurefit_fixtures"]
