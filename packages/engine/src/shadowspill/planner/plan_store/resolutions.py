"""Retained alternatives beside a selected planning result."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from shadowspill.ir import (
    MemorySchedule,
    TaskAlternativeChoice,
)
from shadowspill.store import (
    atomic_text,
    digest_directory,
)

from ..result import ProgramPlanResult, ResolutionPlan
from ..search.toolkit.resolution import recompute_share
from ..serialization import (
    _list,
    _resident_slice_from_value,
    _simulation_result_from_value,
)
from ..serialization.simulation_result import simulation_result_value
from .records import RESOLUTION_SCHEMA
from .request import PlanRequest

if TYPE_CHECKING:
    from . import PlanStore


def resolution_labels(result: ProgramPlanResult) -> dict[str, ResolutionPlan]:
    """Each kept resolution under the name its record is filed by: the share
    of the flexible groups it recomputes, as ``recompute_<share>``, with a
    digest of its selection appended when two resolutions share one."""

    labels: dict[str, ResolutionPlan] = {}
    for plan in result.resolutions:
        share = recompute_share(result.program, plan.selections)
        label = f"recompute_{share.numerator}" + (
            f"of{share.denominator}" if share.denominator != 1 else ""
        )
        if label in labels:
            label += "_" + hashlib.sha256(plan.selection_id.encode()).hexdigest()[:8]
        labels[label] = plan
    return labels


def read_resolutions(
    store: PlanStore, request: PlanRequest
) -> tuple[ResolutionPlan, ...] | None:
    """Restore retained alternatives for consumers that need complete plans."""
    key, program = request.key, request.program
    expected = json.loads(
        json.dumps({**request.boundary(), "schema": RESOLUTION_SCHEMA})
    )
    plans = {}
    directory = digest_directory(store.root, key) / "resolutions"
    try:
        labels = json.loads((directory / "manifest.json").read_text())
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"resolution manifest {directory} cannot be read") from exc
    if (
        not isinstance(labels, list)
        or any(
            not isinstance(label, str)
            or not label
            or label in {".", ".."}
            or Path(label).name != label
            for label in labels
        )
        or len(set(labels)) != len(labels)
    ):
        raise ValueError(f"resolution manifest {directory} has invalid labels")
    for label in labels:
        path = store.resolution_path(key, label)
        try:
            value = json.loads(path.read_text())
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"resolution plan {path} cannot be read") from exc
        if not isinstance(value, dict):
            raise ValueError(f"resolution plan {path} has an invalid schema")
        for field, expected_value in expected.items():
            if value.get(field) != expected_value:
                raise ValueError(f"resolution plan {path} has stale {field} evidence")
        identity = value.get("resolution")
        if (
            not isinstance(identity, dict)
            or identity.get("label") != path.parent.name
            or not isinstance(identity.get("selection_id"), str)
            or not isinstance(identity.get("candidate_id"), str)
        ):
            raise ValueError(f"resolution plan {path} has invalid identity")
        selection_id = identity["selection_id"]
        if selection_id in plans:
            raise ValueError(f"resolution plan {path} repeats {selection_id}")
        selections = tuple(
            TaskAlternativeChoice.from_value(item, f"{path}.selections[{index}]")
            for index, item in enumerate(_list(value.get("selections"), str(path)))
        )
        schedule = MemorySchedule.from_dict(value.get("schedule"))
        schedule.validate(program, selections)
        plans[selection_id] = ResolutionPlan(
            selection_id,
            selections,
            identity["candidate_id"],
            schedule,
            _simulation_result_from_value(
                value.get("simulation_result"), f"{path}.simulation_result"
            ),
            _resident_slice_from_value(
                value.get("resident_slice"), f"{path}.resident_slice"
            ),
        )
        store._record(key, program.digest, path, "read", resolution=True)
    return tuple(plans[key] for key in sorted(plans))


def write_resolutions(
    store: PlanStore,
    request: PlanRequest,
    result: ProgramPlanResult,
) -> None:
    """Replace retained alternatives; publish their completion manifest last."""
    key = request.key
    directory = digest_directory(store.root, key) / "resolutions"
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True, exist_ok=True)
    boundary = request.boundary()
    labeled = resolution_labels(result)
    for label, plan in labeled.items():
        payload: dict[str, object] = {
            **boundary,
            "schema": RESOLUTION_SCHEMA,
            "resolution": {
                "label": label,
                "selection_id": plan.selection_id,
                "candidate_id": plan.candidate_id,
                "recompute_share": str(
                    recompute_share(result.program, plan.selections)
                ),
                "selected": plan.selection_id
                == result.diagnostics.selected_selection_id,
            },
            "schedule": plan.schedule.to_dict(),
            "selections": [item.to_dict() for item in plan.selections],
            "simulation_result": simulation_result_value(plan.simulation),
            "resident_slice": plan.resident_slice.to_dict(),
        }
        path = store.resolution_path(key, label)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_text(path, json.dumps(payload, sort_keys=True, separators=(",", ":")))
        store._record(key, result.program.digest, path, "write", resolution=True)
    # Published last: readers can distinguish complete retention from an
    # interrupted write without making partial artifacts look successful.
    atomic_text(directory / "manifest.json", json.dumps(sorted(labeled)))
