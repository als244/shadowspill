"""Small certified-plan summaries, independent of full plan loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from shadowspill.store import atomic_text

from ..diagnostics import (
    INCUMBENT_CANDIDATE_ID,
    GraphPairOutcome,
    graph_pair_outcomes,
)
from ..diagnostics.plan_summary import PlanSummary, summarize_selected_plan
from ..result import ProgramPlanResult, certified_result
from ..serialization import (
    _boolean,
    _integer,
    _list,
)
from .reader import read_selection
from .records import (
    SUMMARY_SCHEMA,
    PlanSummaryLookup,
    Verdict,
)
from .request import PlanRequest

if TYPE_CHECKING:
    from . import PlanStore


def read_summary(
    store: PlanStore,
    request: PlanRequest,
) -> PlanSummaryLookup | None:
    """Read the small summary or finish interrupted certification."""
    if not store.policy.read_enabled:
        return None
    key, program = request.key, request.program
    path = store.summary_path(key)
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError:
        value = None
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"plan summary {path} cannot be read") from exc
    if value is not None:
        if not isinstance(value, dict) or value.get("schema") != SUMMARY_SCHEMA:
            raise ValueError(f"plan summary {path} has an invalid schema")
        if value.get("key_digest") != key:
            raise ValueError(f"plan summary {path} has the wrong identity")
        if value.get("program_digest") != program.digest:
            raise ValueError(f"plan summary {path} has the wrong ShadowSpillProgram")
        store._record(key, program.digest, path, "read", summary=True)
        return _summary_from_value(value, key, path)
    stored = read_selection(store, request)
    if isinstance(stored, Verdict):
        raise stored.exception()
    if stored is None or stored.certificate is None:
        return None
    return write_summary(
        store, key, certified_result(stored.result, stored.certificate)
    )


def write_summary(
    store: PlanStore, key: str, result: ProgramPlanResult
) -> PlanSummaryLookup:
    """Write what a certified plan promises beside it, and return it.

    The record is read back through the same reader a later call uses,
    so what this call returns is exactly what the store will answer.
    """

    path = store.summary_path(key)
    payload = {
        "schema": SUMMARY_SCHEMA,
        "key_digest": key,
        "program_digest": result.program.digest,
        "schedule_digest": result.schedule.digest,
        "makespan_ns": result.simulation.makespan_ns,
        "answered_with_incumbent": (
            result.diagnostics.selected_candidate_id == INCUMBENT_CANDIDATE_ID
        ),
        "summary": summarize_selected_plan(result).as_dict(),
        "graph_pair_outcomes": [item.as_dict() for item in graph_pair_outcomes(result)],
    }
    if store.policy.write_enabled:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_text(path, json.dumps(payload, sort_keys=True, separators=(",", ":")))
        store._record(key, result.program.digest, path, "write", summary=True)
    return _summary_from_value(payload, key, path)


def _summary_from_value(
    value: dict[str, object], key: str, path: Path
) -> PlanSummaryLookup:
    where = f"plan summary {path}"
    outcomes = _list(value.get("graph_pair_outcomes"), f"{where}.graph_pair_outcomes")
    return PlanSummaryLookup(
        key=key,
        makespan_ns=_integer(value.get("makespan_ns"), f"{where}.makespan_ns"),
        summary=PlanSummary.from_dict(value.get("summary"), f"{where}.summary"),
        graph_pair_outcomes=tuple(
            GraphPairOutcome.from_dict(item, f"{where}.graph_pair_outcomes[{index}]")
            for index, item in enumerate(outcomes)
        ),
        answered_with_incumbent=_boolean(
            value.get("answered_with_incumbent"), f"{where}.answered_with_incumbent"
        ),
    )
