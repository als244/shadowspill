"""Continue interrupted candidate preparation using validated atomic Programs."""

import json
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests.shadowspill.offline._example import program
from workloads.recipes.training import build, build_reuse
from workloads.recipes.training.contracts import BuildSetup


def group():
    return SimpleNamespace(
        rank=0, logical_rank=0, count=1, gather=lambda value: [value]
    )


def test_partial_build_resumes_evaluation_then_skips_both(tmp_path, monkeypatch):
    calls = []
    metadata = {"model": "fixture", "geometry": [2, 17]}
    model = SimpleNamespace(eval=lambda: model)

    @contextmanager
    def setup(options, rank):
        yield BuildSetup(model, None, None, {}, metadata)

    definition = SimpleNamespace(
        setup=setup,
        inputs=lambda options, rank: SimpleNamespace(
            example=lambda: {"args": [(), ()], "kwargs": [{}, {}]}
        ),
    )

    def training(*args, **kwargs):
        calls.append("training")
        return replace(program(), bindings={"kind": "step", "num_examples": 2})

    def evaluation(*args, **kwargs):
        calls.append("evaluation")
        return replace(program(), bindings={"kind": "forward"})

    monkeypatch.setattr(build, "build_step", training)
    monkeypatch.setattr(build, "build_forward", evaluation)
    monkeypatch.setattr(build_reuse, "check_target", lambda program: None)
    candidate = {"path": "fixture/batch-2", "num_examples": 2}

    def invoke():
        return build.build_candidate(
            tmp_path, {"eval_every": 1}, candidate, definition, group()
        )

    assert invoke()
    root = tmp_path / "programs" / candidate["path"]
    training_bytes = (root / "training.ssp").read_bytes()
    (root / "evaluation.ssp").unlink()
    (root / "metadata.json").unlink()
    assert invoke()  # Program survived but metadata/evaluation did not.
    assert (root / "training.ssp").read_bytes() == training_bytes
    assert json.loads((root / "metadata.json").read_text()) == metadata
    mtimes = {p.name: p.stat().st_mtime_ns for p in root.iterdir()}
    assert invoke()
    assert {p.name: p.stat().st_mtime_ns for p in root.iterdir()} == mtimes
    assert calls == ["training", "evaluation", "evaluation"]
    metadata["model"] = "changed"
    with pytest.raises(ValueError, match="adapter metadata differs"):
        invoke()


def test_invalid_target_kind_and_geometry_refuse_reuse(tmp_path, monkeypatch):
    path = tmp_path / "training.ssp"
    replace(program(), bindings={"kind": "step", "num_examples": 2}).save(path)
    monkeypatch.setattr(build_reuse, "check_target", lambda program: None)
    with pytest.raises(ValueError, match="different execution kind"):
        build_reuse.completed(path, group(), kind="forward")
    with pytest.raises(ValueError, match="different example count"):
        build_reuse.completed(path, group(), kind="step", num_examples=4)

    def incompatible(program):
        raise ValueError("target changed")

    monkeypatch.setattr(build_reuse, "check_target", incompatible)
    with pytest.raises(ValueError, match="target changed"):
        build_reuse.completed(path, group(), kind="step", num_examples=2)
    path.write_bytes(b"incomplete archive")
    with pytest.raises(ValueError, match="cannot reuse completed build"):
        build_reuse.completed(path, group(), kind="step", num_examples=2)


def test_collective_missing_program_and_peer_error_are_shared(tmp_path, monkeypatch):
    path = tmp_path / "training.ssp"
    replace(program(), bindings={"kind": "step"}).save(path)
    monkeypatch.setattr(build_reuse, "check_target", lambda program: None)
    missing = SimpleNamespace(gather=lambda value: [value, False])
    assert not build_reuse.completed(path, missing, kind="step")
    peer_error = SimpleNamespace(
        gather=lambda value: [value, True if value is True else "peer target mismatch"]
    )
    with pytest.raises(ValueError, match="peer target mismatch"):
        build_reuse.completed(path, peer_error, kind="step")
