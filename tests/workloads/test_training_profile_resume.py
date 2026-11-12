"""Interrupted profiling preserves complete evidence and coordinates reuse."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from orchestration.workers import Workers
from shadowspill import ProfiledProgram, ProfilingOptions
from tests.shadowspill.offline._example import profiles, program
from workloads.recipes.training import profile, profile_reuse


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    model = replace(program(), setup={"gpu": "fixture"})
    measured = profiles()
    measured.setup = {"gpu": "fixture"}
    measured.metadata.update(memory_kind="measured", external_headroom_bytes=0)
    measured.metadata["profiling_options"] = ProfilingOptions().to_dict()
    candidate = {"path": "candidate/batch-1"}
    options = {"eval_every": 1, "external_headroom_mib": 0}
    group = Workers(0, {0: 0})
    for kind in ("training", "evaluation"):
        model.save(tmp_path / "programs" / candidate["path"] / f"{kind}.ssp")
    path = tmp_path / "profiles" / candidate["path"] / "training.ssp"
    measured.save(path)
    monkeypatch.setattr(profile_reuse, "check_target", lambda value: None)
    return SimpleNamespace(
        root=tmp_path,
        model=model,
        measured=measured,
        path=path,
        candidate=candidate,
        options=options,
        group=group,
    )


def test_profile_resume_recreates_applied_artifact_and_measures_only_missing(
    prepared, monkeypatch
):
    p = prepared
    stamp = p.path.stat().st_mtime_ns
    measured = []

    def measure(model, **kwargs):
        measured.append(model.digest)
        return p.measured

    monkeypatch.setattr(profile, "measure_program", measure)
    profile.profile_candidate(p.root, p.options, p.candidate, p.group, {})
    assert measured == [p.model.digest]
    assert p.path.stat().st_mtime_ns == stamp
    for kind in ("training", "evaluation"):
        value = ProfiledProgram.load(
            p.root / "profiled" / p.candidate["path"] / f"{kind}.ssp"
        )
        assert value.program.digest == p.model.digest
    profile.profile_candidate(p.root, p.options, p.candidate, p.group, {})
    assert len(measured) == 1
    profile.profile_candidate(
        p.root, {**p.options, "refresh_profiles": True}, p.candidate, p.group, {}
    )
    assert len(measured) == 3


@pytest.mark.parametrize("change", ["setup", "implementation", "headroom", "estimated"])
def test_completed_profile_refuses_incompatible_evidence(prepared, change):
    p = prepared
    if change == "setup":
        p.measured.setup["gpu"] = "other"
    elif change == "implementation":
        key = "update_impl"
        p.measured.tasks[key] = replace(
            p.measured.tasks[key], compatibility_digest="other"
        )
    elif change == "headroom":
        p.measured.metadata["external_headroom_bytes"] = 1
    else:
        p.measured.metadata["memory_kind"] = "estimated"
    p.measured.save(p.path)
    with pytest.raises(ValueError, match="cannot reuse completed profile"):
        profile_reuse.completed(p.path, p.model, p.options, p.group)


def test_collective_reuse_agrees_on_missing_files_and_peer_target_failure(prepared):
    p = prepared
    missing = SimpleNamespace(gather=lambda present: [present, False])
    assert profile_reuse.completed(p.path, p.model, p.options, missing) is None
    peer_failure = SimpleNamespace(
        gather=lambda value: (
            [True, True] if isinstance(value, bool) else [value, "peer target differs"]
        )
    )
    with pytest.raises(ValueError, match="peer target differs"):
        profile_reuse.completed(p.path, p.model, p.options, peer_failure)


def test_target_validation_precedes_reuse(prepared, monkeypatch):
    def wrong_target(model):
        raise ValueError("target differs")

    monkeypatch.setattr(profile_reuse, "check_target", wrong_target)
    p = prepared
    with pytest.raises(ValueError, match="target differs"):
        profile_reuse.completed(p.path, p.model, p.options, p.group)


def test_changed_timing_policy_requires_fresh_measurement(prepared):
    p = prepared
    changed = {**p.options, "profiling": {"maximum_samples": 7}}
    assert profile_reuse.completed(p.path, p.model, changed, p.group) is None
