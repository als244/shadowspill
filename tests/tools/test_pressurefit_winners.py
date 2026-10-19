"""Winner exports preserve the executed schedule and physical certificate."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from shadowspill.planner.annotated_plan import AnnotatedProgramPlan
from shadowspill.planner.diagnostics import INCUMBENT_CANDIDATE_ID, IncumbentDiagnostic
from tests.tools.test_pressurefit_matrix import collection
from tools.pressurefit_trace.data import Trajectory
from tools.pressurefit_trace.plan.data import PlanData
from tools.pressurefit_trace.winner_capture import annotated_selection
from tools.pressurefit_trace.winners import prepare, trace_origin


def test_executed_winner_roundtrip_and_trace(tmp_path):
    planning = collection(tmp_path)
    plan = AnnotatedProgramPlan.from_json((planning / "program.plan.json").read_text())
    admission = SimpleNamespace(
        layout=plan.fixed_layout,
        simulator_input=plan.simulation_admission,
        simulation=plan.simulation,
    )
    exported = annotated_selection(
        plan.result,
        plan.effective_facts,
        admission,
        execution_bytes=plan.memory_budgets.execution_bytes,
        spill_bytes=plan.memory_budgets.spill_bytes,
        attempts=plan.attempts,
    )
    restored = AnnotatedProgramPlan.from_json(exported.to_json())
    assert restored.fixed_layout.digest == plan.fixed_layout.digest
    assert restored.result.schedule.digest == plan.result.schedule.digest
    root = tmp_path / "quickstart" / "rank-00000"
    budget = plan.memory_budgets.execution_bytes
    bundle = root / "winners" / f"{budget}bytes"
    bundle.mkdir(parents=True)
    for name in ("plan.json", "search-plan.json"):
        (bundle / name).write_text(exported.to_json())
    (bundle / "search-logical.json").write_text("{}")
    (bundle / "semantics.json").write_text("{}")
    (bundle / "bundle.json").write_text(
        json.dumps(
            {
                "schedule_digest": plan.result.schedule.digest,
                "capture_identity": "example",
            }
        )
    )
    (root / "executed").mkdir()
    execution = {
        "status": "executed",
        "capture_identity": "example",
        "candidate": "1",
        "ordering": {"depth": 1, "breadth": 1},
        "ordering_label": "1x1rp",
        "execution_budget_bytes": budget,
        "requested_execution_budget_bytes": [budget],
        "spill_budget_bytes": plan.memory_budgets.spill_bytes,
        "median_seconds": 1.0,
    }
    record = root / "executed" / f"{budget}bytes.json"
    record.write_text(json.dumps(execution))
    output = tmp_path / "videos"
    manifest = prepare(root.parent, output, workload="Example")
    assert manifest["complete"] and len(manifest["cases"]) == 1
    case = output / manifest["cases"][0]["path"]
    data = PlanData.load(case / "plan.json", metadata=case / "metadata.json")
    assert data.schedule_digest == plan.result.schedule.digest
    trace = Trajectory.load(case / manifest["cases"][0]["trace"])
    trace.trace.verify()
    assert prepare(root.parent, output, workload="Example") == manifest
    execution["capture_identity"] = "different"
    record.write_text(json.dumps(execution))
    with pytest.raises(ValueError, match="does not identify"):
        prepare(root.parent, output)


def test_carried_winner_uses_original_budget_search(tmp_path):
    planning = collection(tmp_path)
    original = AnnotatedProgramPlan.from_json(
        (planning / "program.plan.json").read_text()
    )
    selected = next(
        row
        for row in original.result.diagnostics.resolved_programs
        if row.selection_id == original.result.diagnostics.selected_selection_id
    )
    diagnostics = replace(
        original.result.diagnostics,
        selected_candidate_id=INCUMBENT_CANDIDATE_ID,
        resolved_programs=(
            replace(
                selected,
                candidate_evaluations=(),
                selected_candidate_id=INCUMBENT_CANDIDATE_ID,
                incumbent=IncumbentDiagnostic(
                    "valid",
                    selected.selected_makespan_ns,
                    None,
                    True,
                    original.result.schedule.digest,
                ),
            ),
        ),
    )
    carried = replace(
        original, result=replace(original.result, diagnostics=diagnostics)
    )
    with pytest.raises(ValueError, match="no matching original search"):
        trace_origin(carried, tmp_path)
    archive = tmp_path / "search-winners"
    archive.mkdir()
    origin_path = archive / "earlier.plan.json"
    origin_path.write_text(original.to_json())
    found, logical, path = trace_origin(carried, tmp_path)
    assert path == origin_path
    assert found.result.schedule.digest == carried.result.schedule.digest
    assert logical.diagnostics.selected_candidate_id != INCUMBENT_CANDIDATE_ID
