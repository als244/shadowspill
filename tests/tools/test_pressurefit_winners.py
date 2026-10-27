"""Winner videos read current Plans without model imports or export hooks."""

import json
import os
import subprocess
import sys
from dataclasses import replace

import pytest

from benchmarking.quickstart.phases.common import plan_identity, read, write
from benchmarking.quickstart.phases.search import search_options
from shadowspill import MemoryBudgets, Plan, plan_program
from tests.benchmarking._profiled import measured
from tests.shadowspill.offline._example import communication
from tools.pressurefit_trace.data import Trajectory
from tools.pressurefit_trace.plan.data import PlanData
from tools.pressurefit_trace.plan.placement import PlacementReplay
from tools.pressurefit_trace.winners import prepare
from tools.pressurefit_trace.winners.plan import diagnostic_plan


@pytest.fixture
def current_winner(tmp_path):
    root = tmp_path / "quickstart"
    request = {"factory": "must_not_import:factory", "search_workers": 1}
    plan = plan_program(
        measured(),
        memory_budgets=MemoryBudgets(4096, 4096),
        communication_info=communication(),
        search_options=search_options(request),
    )
    winner = {
        "file": "budget-0000.ssp",
        "budget": 4096,
        "candidate": "sample",
        "ordering": "default",
        "plan_digest": plan_identity(plan),
    }
    plan.save(root / "plans" / winner["file"])
    write(root / "plans/index.json", [winner])
    write(root / "programs/index.json", {"world_size": 1})
    write(root / "request.json", request)
    write(
        root / "execution/budget-0000/result.json",
        {
            **winner,
            "median_seconds": 1,
            "per_rank": [{"logical_rank": 0, "steps": [{"step": 1, "seconds": 1}]}],
        },
    )
    return root


def test_current_winner_videos_preserve_certificate_and_cut_history(
    current_winner, tmp_path
):
    output = tmp_path / "videos"
    script = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'must_not_import'} or any(
            fullname == name or fullname.startswith(name + '.')
            for name in ('shadowspill.build', 'shadowspill.execution',
                         'shadowspill.pytorch', 'benchmarking.quickstart.runner')
        ):
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from tools.pressurefit_trace.winners.__main__ import main
main(sys.argv[1:])
"""
    child = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            script,
            str(current_winner),
            "--output",
            str(output),
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    manifest = read(output / "matrix.json")
    assert manifest["complete"] and len(manifest["cases"]) == 1
    entry = manifest["cases"][0]
    case = output / entry["path"]
    saved = Plan.load(current_winner / "plans/budget-0000.ssp")
    view = PlanData.load(case / "plan.json", metadata=case / "metadata.json")
    assert view.schedule_digest == saved.schedule.digest
    assert view.layout_digest == saved.memory_layout.digest
    assert view.end_ns == saved.timeline.makespan_ns
    assert view.resident_aliases == {"weight_storage"}
    PlacementReplay(view).verify()
    trace = Trajectory.load(case / entry["trace"])
    trace.trace.verify()
    assert (
        read(case / entry["trace"] / "manifest.json")["answer"]["schedule_digest"]
        == saved.schedule.digest
    )
    assert (
        read(case / "semantics.json")["objects"]["weight"]["label"] == "linear.weight"
    )
    assert entry["measured_seconds"] == 1
    assert prepare(current_winner, output) == manifest
    execution = current_winner / "execution/budget-0000/result.json"
    write(execution, {**read(execution), "median_seconds": 2})
    with pytest.raises(ValueError, match="winner changed"):
        prepare(current_winner, output)


def test_projection_handles_no_resident_slice_and_requires_certified_memory():
    options = search_options({"search_workers": 1})
    options = replace(
        options, generic=replace(options.generic, minimum_object_bytes_evict_eligible=0)
    )
    plan = plan_program(
        measured(),
        memory_budgets=MemoryBudgets(4096, 4096),
        communication_info=communication(),
        search_options=options,
    )
    annotated, _ = diagnostic_plan(plan, options)
    assert annotated.result.resident_slice.aliases == ()
    PlanData(json.loads(annotated.to_json()))
    with pytest.raises(ValueError, match="certified physical layout"):
        diagnostic_plan(
            replace(plan, memory_layout=None, _admission_facts=None), options
        )
