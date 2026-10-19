"""Every resolution keeps its own schedule, trace answer, and certificate."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from types import SimpleNamespace

import pytest

from shadowspill.planner import plan_program
from shadowspill.planner.annotated_plan import AnnotatedProgramPlan
from shadowspill.planner.program_inputs import ShadowSpillPlanningProblem
from tests.shadowspill.planner.test_plan_store import (
    FEW_CANDIDATES,
    _alternatives_program,
    _facts,
    config,
)
from tools.pressurefit_trace.data import Trajectory
from tools.pressurefit_trace.matrix import prepare
from tools.pressurefit_trace.plan.data import PlanData
from tools.pressurefit_trace.plan.placement import PlacementReplay
from tools.pressurefit_trace.render_batch import render_batch


def collection(tmp_path):
    program = _alternatives_program()
    machine = config()
    capacity = machine.devices[0].capacity_bytes
    problem = ShadowSpillPlanningProblem(
        role="step",
        program=program,
        initial_residency=(),
        final_residency=(),
        simulation_config=machine,
        admission_facts=_facts(program, capacity),
        source_execution_budget_bytes=capacity,
        maximum_execution_budget_bytes=capacity,
        maximum_spill_budget_bytes=machine.spill_capacity_bytes,
        fixed_execution_bytes=0,
        object_reserve_bytes=0,
        dynamic_scratch_reserve_bytes=0,
    )
    options = replace(
        FEW_CANDIDATES, generic=replace(FEW_CANDIDATES.generic, deterministic=True)
    )
    selected = plan_program(problem, search_options=options, verbose=False)
    dump, planning = tmp_path / "dump", tmp_path / "plans"
    dump.mkdir()
    planning.mkdir()
    (dump / "program.problem.json").write_text(problem.to_json())
    (dump / "program.semantics.json").write_text("{}")
    (dump / "dump.json").write_text("{}")
    (planning / "program.plan.json").write_text(selected.to_json())
    (planning / "plans.json").write_text(
        json.dumps(
            {
                "source": str(dump),
                "execution_budget_bytes": capacity,
                "physical_budget_bytes": capacity,
                "spill_budget_bytes": machine.spill_capacity_bytes,
                "programs": [
                    {
                        "name": "program",
                        "problem": "program.problem.json",
                        "semantics": "program.semantics.json",
                        "geometry": "1",
                        "ordering": "1x1rp",
                    }
                ],
            }
        )
    )
    return planning


def test_matrix_keeps_both_resolutions_and_resumes_without_search(
    tmp_path, monkeypatch
):
    planning = collection(tmp_path)
    root = tmp_path / "matrix"
    matrix = prepare(planning, root, workers=1)
    assert matrix["complete"]
    assert len(matrix["cases"]) == 2
    assert {item["makespan_ns"] for item in matrix["cases"]} == {15, 25}
    for item in matrix["cases"]:
        assert item["status"] == "ready"
        case = root / item["path"]
        plan = AnnotatedProgramPlan.from_json((case / "plan.json").read_text())
        assert plan.simulation.makespan_ns == item["makespan_ns"]
        assert (
            plan.result.diagnostics.selected_selection_id
            == item["fingerprint"]["selection_id"]
        )
        data = PlanData.load(case / "plan.json", metadata=case / "metadata.json")
        PlacementReplay(data).verify()
        trajectory = Trajectory.load(case / item["trace"])
        trajectory.trace.verify()
        assert (
            json.loads((case / item["trace"] / "manifest.json").read_text())["answer"][
                "schedule_digest"
            ]
            == plan.result.schedule.digest
        )

    def refuse(*args, **kwargs):
        raise AssertionError("completed matrix must not search again")

    monkeypatch.setattr("tools.pressurefit_trace.matrix.plan_program", refuse)
    assert prepare(planning, root, workers=1) == matrix


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_batch_encodes_publishes_and_resumes_verified_movies(tmp_path, monkeypatch):
    planning = collection(tmp_path)
    matrix = tmp_path / "matrix"
    prepare(planning, matrix, workers=1)
    output, published = tmp_path / "videos", tmp_path / "library"
    args = SimpleNamespace(
        matrix=matrix,
        output=output,
        publish_to=published,
        title="Generic example",
        videos=["execution"],
        resolution="720p",
        quality="preview",
        fps=5,
        motion_fps=5,
        jobs=1,
        emission_timing=(0.2,) * 5,
        execution_timing=(0.2,) * 5,
        cut_options_json="{}",
    )
    render_batch(args)
    progress = json.loads((output / "progress.json").read_text())
    assert progress["total"] == 2
    assert all(row["status"] == "complete" for row in progress["movies"])
    for row in progress["movies"]:
        assert row["validation"]["decode"] == "passed"
        assert (published / row["relative"]).read_bytes() == (
            output / row["relative"]
        ).read_bytes()
    before = {p: p.stat().st_mtime_ns for p in output.rglob("*.mp4")}
    monkeypatch.setattr(
        "tools.pressurefit_trace.render_batch.movie_command",
        lambda *_: pytest.fail("completed movie was rendered again"),
    )
    render_batch(args)
    assert {p: p.stat().st_mtime_ns for p in output.rglob("*.mp4")} == before
