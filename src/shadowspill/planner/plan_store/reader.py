"""Validate and restore current planning-cache records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from shadowspill.ir import (
    MemorySchedule,
    TaskAlternativeChoice,
)

from ..diagnostics import PlanningDiagnostics
from ..result import ProgramPlanResult
from ..serialization import (
    _mapping,
    _resident_slice_from_value,
    _simulation_result_from_value,
)
from ..serialization.certificate_file import (
    CERTIFICATE_FILE,
    read_certificate,
    read_certificate_meta,
)
from ..serialization.simulation_admission import simulation_admission_value
from ..serialization.simulation_result import simulation_result_value
from .records import (
    SCHEMA,
    PlanLookup,
    Verdict,
)
from .request import PlanRequest
from .resolutions import read_resolutions

if TYPE_CHECKING:
    from . import PlanStore


def read_payload(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"planned program {path} cannot be read") from exc
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError(f"planned program {path} has an invalid schema")
    return value


def read_selection(
    store: PlanStore,
    request: PlanRequest,
    *,
    keep_resolutions: bool = False,
) -> PlanLookup | Verdict | None:
    """Validate identity, request evidence and schedule without re-simulation."""
    key, program = request.key, request.program
    path = store.path(key)
    value = read_payload(path)
    if value is None:
        return None
    expected = request.boundary()
    normalized = json.loads(json.dumps(expected, sort_keys=True, separators=(",", ":")))
    if value.get("key_digest") != key:
        raise ValueError(f"planned program {path} has the wrong identity")
    if value.get("program_digest") != program.digest:
        raise ValueError(f"planned program {path} has the wrong ShadowSpillProgram")
    for field, expected_value in normalized.items():
        if value.get(field) != expected_value:
            raise ValueError(f"planned program {path} has stale {field} evidence")
    verdict = value.get("verdict")
    if verdict is not None:
        if not isinstance(verdict, dict):
            raise ValueError(f"planned program {path} has an invalid verdict")
        store._record(key, program.digest, path, "read")
        return Verdict(
            outcome=str(verdict.get("outcome")),
            error=str(verdict.get("error")),
            kind=None if verdict.get("kind") is None else str(verdict.get("kind")),
            message=str(verdict.get("message")),
        )
    schedule = MemorySchedule.from_dict(value.get("schedule"))
    raw_selections = value.get("selections")
    if not isinstance(raw_selections, list):
        raise ValueError(f"planned program {path} has invalid selections")
    selections = tuple(
        TaskAlternativeChoice.from_value(item, f"cache.selections[{index}]")
        for index, item in enumerate(raw_selections)
    )
    schedule.validate(program, selections)
    simulation = _simulation_result_from_value(
        value.get("simulation_result"), f"{path}.simulation_result"
    )
    diagnostics = _diagnostics_from_value(value.get("diagnostics"), path)
    if diagnostics.selected_makespan_ns != simulation.makespan_ns:
        raise ValueError(f"planned program {path} has inconsistent simulator evidence")
    retained = read_resolutions(store, request) if keep_resolutions else ()
    if retained is None:
        # The process may have stopped while writing alternative records.
        # Treat incomplete retention as a cache miss, never a partial answer.
        return None
    result = ProgramPlanResult(
        program=program,
        search_options=request.options,
        initial_residency=request.initial_residency,
        final_residency=request.final_residency,
        simulation_config=request.config,
        schedule=schedule,
        selections=selections,
        simulation=simulation,
        diagnostics=diagnostics,
        resident_slice=_resident_slice_from_value(
            value.get("resident_slice"), f"{path}.resident_slice"
        ),
        admission_facts=request.admission,
        resolutions=retained,
    )
    certificate_path = store.certificate_path(key)
    certificate = (
        read_certificate(certificate_path) if certificate_path.exists() else None
    )
    store._record(key, program.digest, path, "read")
    return PlanLookup(result, True, key, certificate)


def _diagnostics_from_value(value: object, path: Path) -> PlanningDiagnostics:
    try:
        return PlanningDiagnostics.from_value(value, "cache.diagnostics")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"planned program {path} has invalid diagnostics") from exc


def read_stored_selection(path: Path, *, layout: bool = True) -> dict[str, object]:
    """Read current selection JSON and, when present, its separate certificate.

    ``layout=False`` reads certificate metadata without the layout arrays.
    Uncertified selections return only their selection record.
    """

    value: dict[str, object] = _mapping(json.loads(path.read_text()), str(path))
    certificate = path.parent / CERTIFICATE_FILE
    if not certificate.exists():
        return value
    if not layout:
        meta = read_certificate_meta(certificate)
        value["admission_certificate"] = {
            "facts_digest": meta["facts_digest"],
            "layout": meta["layout"],
        }
        return value
    admission = read_certificate(certificate)
    value["admission_certificate"] = {
        "facts_digest": admission.layout.facts_digest,
        "layout": admission.layout.to_dict(),
        "simulator_input": simulation_admission_value(admission.simulator_input),
        "simulation": simulation_result_value(admission.simulation),
    }
    return value
