"""A recoverable candidate failure survives phase boundaries and can be retried."""

import weakref
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from benchmarking.quickstart.phases import build, communicate, plan, profile
from benchmarking.quickstart.phases.common import paths, read, write
from shadowspill.artifacts.sweep import SweepResult
from tests.shadowspill.offline._example import profiles, program


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    request = {
        "output_dir": str(tmp_path),
        "device": "cuda:0",
        "warmup_seconds": 0.001,
        "measure_seconds": 0.002,
        "external_headroom_bytes": 0,
        "search_budgets": [200, 300],
        "spill_bytes": 64,
        "search_workers": 1,
        "plots": False,
    }
    setup = {
        "model_factory": lambda: SimpleNamespace(parameters=lambda: ()),
        "optimizer": lambda parameters: None,
        "objective": None,
        "candidates": {name: {"args": [(name,)]} for name in ("large", "fits")},
    }
    monkeypatch.setattr(build, "experiment", lambda request: setup)
    monkeypatch.setattr(
        build,
        "build_step",
        lambda _, **kw: replace(program(), name=kw["example_calls"][0][0][0]),
    )
    return request


@pytest.mark.parametrize("phase", ["build", "profile"])
def test_oom_releases_temporaries_and_continues(prepared, monkeypatch, phase):
    visited, held = [], []

    def compute(name):
        visited.append(name)
        if name == "large":
            temporary = torch.ones(8)
            held.append(weakref.ref(temporary))
            try:
                raise torch.OutOfMemoryError("candidate exceeds device capacity")
            except torch.OutOfMemoryError as error:
                raise RuntimeError("wrapped task failure") from error
        assert held[0]() is None
        return replace(program(), name=name) if phase == "build" else profiles()

    if phase == "build":
        monkeypatch.setattr(
            build, "build_step", lambda _, **kw: compute(kw["example_calls"][0][0][0])
        )
        build.run(prepared)
    else:
        build.run(prepared)
        monkeypatch.setattr(profile, "measure_program", lambda p, **kw: compute(p.name))
        profile.run(prepared)
    root, index = paths(prepared, "programs" if phase == "build" else "profiled")
    assert visited == ["large", "fits"]
    failed, complete = index["candidates"]
    assert failed["failure"]["phase"] == phase
    assert failed["failure"]["type"] == "OutOfMemoryError"
    assert "candidate exceeds device capacity" in failed["failure"]["message"]
    assert "failure" not in complete
    assert (
        root / ("programs" if phase == "build" else "profiled") / complete["file"]
    ).is_file()
    if phase == "profile":
        # A profile rerun retries the built Program; it does not inherit the
        # prior profiling failure or mistake an older archive for a success.
        assert "failure" not in read(root / "programs/index.json")["candidates"][0]
        monkeypatch.setattr(profile, "measure_program", lambda *a, **kw: profiles())
        profile.run(prepared)
        assert all(
            "failure" not in row
            for row in read(root / "profiled/index.json")["candidates"]
        )


@pytest.mark.parametrize("phase", ["build", "profile"])
def test_other_errors_stop_and_invalidate_old_phase_index(prepared, monkeypatch, phase):
    def fail(*args, **kwargs):
        raise ValueError("invalid task contract")

    build.run(prepared)
    root, _ = paths(prepared)
    write(root / "profiled/index.json", {"candidates": ["old success"]})
    module, operation = (
        (build, "build_step") if phase == "build" else (profile, "measure_program")
    )
    monkeypatch.setattr(module, operation, fail)
    with pytest.raises(ValueError, match="invalid task contract"):
        module.run(prepared)
    assert not (root / "profiled/index.json").exists()


def test_all_failed_candidates_still_produce_diagnostic_report(prepared, monkeypatch):
    def fail(*args, **kwargs):
        raise torch.OutOfMemoryError("too large to build")

    monkeypatch.setattr(build, "build_step", fail)
    build.run(prepared)
    profile.run(prepared)
    communicate.run(prepared)
    plan.run(prepared)
    root, _ = paths(prepared)
    result = SweepResult.load(root / "planning/sweep.ssp")
    assert len(result.cases) == 4
    assert all(
        case.plan is None and case.failure["phase"] == "build" for case in result.cases
    )
    assert read(root / "plans/index.json") == []
    assert "too large to build" in (root / "report/index.html").read_text()
    assert not (root / "communication.ssp").exists()
