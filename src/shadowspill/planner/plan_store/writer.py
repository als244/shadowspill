"""Write selected plans and cached search refusals."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from shadowspill.errors import PlanSearchExhaustedError
from shadowspill.store import atomic_text

from ..diagnostics.json import without_measurements
from ..result import ProgramPlanResult
from ..serialization.simulation_result import simulation_result_value
from .reader import read_payload
from .request import PlanRequest, incumbent_field
from .resolutions import write_resolutions

if TYPE_CHECKING:
    from . import PlanStore


def write_selection(
    store: PlanStore,
    request: PlanRequest,
    result: ProgramPlanResult,
    incumbent: ProgramPlanResult | None = None,
    improve: bool = False,
    keep_resolutions: bool = False,
) -> None:
    if not store.policy.write_enabled:
        return
    key = request.key
    path = store.path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        **request.boundary(),
        **incumbent_field(incumbent),
        "schedule": result.schedule.to_dict(),
        "selections": [item.to_dict() for item in result.selections],
        "simulation_result": simulation_result_value(result.simulation),
        "diagnostics": result.diagnostics.to_dict(),
        "resident_slice": result.resident_slice.to_dict(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    existing = None if store.policy.overwrite or improve else read_payload(path)
    # A successful answer can replace a cached refusal.
    if existing is not None and "verdict" not in existing:
        # Provenance is not the answer: the same plan found with or
        # without a plan in hand is the same plan.
        if _answer(existing) != _answer(payload):
            raise ValueError(
                "a fresh search differs from the stored planned program; "
                "use a 'refresh' store mode or a new export_bypass_key: "
                f"{path}"
            )
        store._record(key, result.program.digest, path, "matched")
    else:
        # A summary describes the plan it was written beside; a new plan
        # has none until it is certified.
        store.summary_path(key).unlink(missing_ok=True)
        store.certificate_path(key).unlink(missing_ok=True)
        atomic_text(path, encoded)
        store._record(
            key, result.program.digest, path, "improved" if improve else "write"
        )
    if keep_resolutions:
        write_resolutions(store, request, result)


def write_verdict(
    store: PlanStore,
    request: PlanRequest,
    error: BaseException,
) -> None:
    if not store.policy.write_enabled:
        return
    key = request.key
    path = store.path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        **request.boundary(),
        "verdict": {
            "outcome": (
                "exhausted"
                if isinstance(error, PlanSearchExhaustedError)
                else "infeasible"
            ),
            "error": type(error).__name__,
            "kind": getattr(error, "kind", None),
            "message": str(error),
        },
    }
    # A verdict supersedes the plan the key held, and with it what was
    # written beside that plan.
    store.summary_path(key).unlink(missing_ok=True)
    store.certificate_path(key).unlink(missing_ok=True)
    atomic_text(path, json.dumps(payload, sort_keys=True, separators=(",", ":")))
    store._record(key, request.program.digest, path, "verdict")


def _answer(payload: object) -> object:
    """A stored record as an answer: without measurements or the plan to beat."""

    value = without_measurements(payload)
    if isinstance(value, dict):
        return {name: item for name, item in value.items() if name != "incumbent"}
    return value
