"""Explicit chunk candidates use the measured, budget-constrained plan search."""

from types import SimpleNamespace

import pytest

from benchmarks.quickstart.phases import build, plan
from benchmarks.quickstart.phases.candidates import example_inputs
from benchmarks.quickstart.phases.common import read, write
from benchmarks.quickstart.phases.distributed import preparation_size
from tests.shadowspill.offline._example import communication, program


def test_candidate_build_options_override_defaults_without_changing_inputs(
    tmp_path, monkeypatch
):
    shared = {
        "partition": object(),
        "chunking": object(),
        "graph_pair_policy": object(),
    }
    override, retention = object(), object()
    calls = []
    setup = {
        "model_factory": lambda: SimpleNamespace(parameters=lambda: ()),
        "optimizer": lambda _: None,
        "objective": None,
        "build_options": shared,
        "candidates": {
            "default": {"args": [(1,)]},
            "custom": {
                "kwargs": [{"features": 2}],
                "build_options": {
                    "chunking": override,
                    "graph_pair_policy": retention,
                },
            },
            "unchunked": {
                "args": [(3,)],
                "build_options": {"chunking": None},
            },
        },
    }

    def capture(_, **options):
        calls.append(options)
        return program()

    monkeypatch.setattr(build, "experiment", lambda request: setup)
    monkeypatch.setattr(build, "build_step", capture)
    build.run({"output_dir": str(tmp_path), "device": "cuda:0"})
    assert [call["chunking"] for call in calls] == [shared["chunking"], override, None]
    assert [call["graph_pair_policy"] for call in calls] == [
        shared["graph_pair_policy"],
        retention,
        shared["graph_pair_policy"],
    ]
    assert all(call["partition"] is shared["partition"] for call in calls)
    assert calls[1]["example_calls"] == [((), {"features": 2})]
    assert example_inputs(setup["candidates"]["custom"]) == {
        "kwargs": [{"features": 2}]
    }
    assert setup["candidates"]["custom"]["build_options"] == {
        "chunking": override,
        "graph_pair_policy": retention,
    }
    assert [
        row["name"] for row in read(tmp_path / "programs/index.json")["candidates"]
    ] == ["default", "custom", "unchunked"]


@pytest.mark.parametrize("key", ["resources", "parallelism"])
def test_candidate_topology_is_refused_before_selecting_workers(key):
    setup = {"candidates": {"chunk4": {"build_options": {key: None}}}}
    with pytest.raises(ValueError, match="shared worker topology"):
        preparation_size(setup)


@pytest.mark.parametrize(
    "candidate, message",
    [
        ({"args": [(1,)], "chunk_size": 4}, "unknown candidate entries"),
        ({"build_options": {"chunking": None}}, "nonempty"),
        ({"args": [(1,)], "kwargs": [{}, {}]}, "equally sized"),
    ],
)
def test_invalid_candidate_is_refused_before_model_construction(
    tmp_path, monkeypatch, candidate, message
):
    monkeypatch.setattr(
        build, "experiment", lambda request: {"candidates": {"invalid": candidate}}
    )
    with pytest.raises(ValueError, match=message):
        build.run({"output_dir": str(tmp_path), "device": "cuda:0"})


def test_each_budget_selects_fastest_admitted_profiled_candidate(tmp_path):
    from shadowspill import Profiles, SweepResult, TaskProfile, apply_profiles

    entries = []
    for name, duration, workspace in [("chunk2", 3e-6, 16), ("chunk8", 1e-6, 160)]:
        filename = name + ".ssp"
        measured = Profiles(
            tasks={
                "update_impl": TaskProfile(duration, workspace_bytes=workspace),
                "boundary": TaskProfile(0),
            },
            fixed_device_bytes=20,
        )
        apply_profiles(program(), measured).save(tmp_path / "profiled" / filename)
        entries.append({"name": name, "file": filename})
    entries.append(
        {
            "name": "chunk32",
            "file": "missing.ssp",
            "num_examples": 1,
            "failure": {
                "type": "OutOfMemoryError",
                "phase": "profile",
                "message": "too large to measure",
            },
        }
    )
    write(tmp_path / "profiled/index.json", {"candidates": entries})
    communication().save(tmp_path / "communication.ssp")
    plan.run(
        {
            "output_dir": str(tmp_path),
            "search_budgets": [200, 400],
            "spill_bytes": 64,
            "search_workers": 1,
            "plots": False,
        }
    )
    winners = read(tmp_path / "plans/index.json")
    assert [(row["budget"], row["candidate"]) for row in winners] == [
        (200, "chunk2"),
        (400, "chunk8"),
    ]
    result = SweepResult.load(tmp_path / "planning/sweep.ssp")
    assert len(result.cases) == 6
    large_at_small_budget = next(
        case
        for case in result.cases
        if case.candidate.startswith("chunk8:")
        and case.memory_budgets.execution_bytes == 200
    )
    assert large_at_small_budget.plan is None and large_at_small_budget.failure
    assert sum(case.failure.get("phase") == "profile" for case in result.cases) == 2
