"""Preparation owns one directory and never repairs incompatible saved results."""

import subprocess
import sys
import zipfile
from dataclasses import replace

import pytest

from engine_tests.training.test_setup_resolution import installed
from engine_tests.training.test_training_bundle import make_bundle
from shadowspill.training.preparation.phases import job
from shadowspill_client import Memory


@pytest.fixture
def prepared(monkeypatch):
    bundle, checked = make_bundle(monkeypatch)
    monkeypatch.setattr(job, "load_registry", lambda packages: installed())
    launches = []

    def launch(package, root, selected, **options):
        launches.append((package, root, selected, options))
        bundle.save(root / job.BUNDLE_NAME)

    monkeypatch.setattr(job, "run_phases", launch)
    return bundle, checked, launches


def test_job_roundtrip_defers_full_recheck_to_execution_admission(prepared, tmp_path):
    bundle, checked, launches = prepared
    root = tmp_path / "job"
    args = {"packages": ["external-classifier-fixture"], "config": bundle.config}
    first = job.prepare_job(root, **args)
    assert first.to_dict() == bundle.to_dict()
    saved = job.PreparationJob.load(root / job.JOB_NAME)
    assert saved.config == bundle.config
    assert saved.packages == tuple(args["packages"])
    assert launches == [
        (
            "shadowspill.training.preparation.phases",
            root,
            "all",
            {
                "ranks": 1,
                "preparation_ranks": 1,
                "phases": ("build", "profile", "communicate", "plan"),
                "environment": None,
                "launch": None,
            },
        )
    ]
    assert checked == []
    assert job.prepare_job(root, **args).to_dict() == first.to_dict()
    assert len(launches) == 1
    assert checked == []
    first.recheck(installed())
    assert len(checked) == len(bundle.plans[0])


def test_job_preserves_unrelated_directories_and_rejects_changed_setup(
    prepared, tmp_path
):
    bundle, _, launches = prepared
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "keep.txt").write_text("unrelated work")
    args = {"packages": ["external-classifier-fixture"], "config": bundle.config}
    with pytest.raises(ValueError, match="no matching preparation job"):
        job.prepare_job(unrelated, **args)
    assert sorted(p.name for p in unrelated.iterdir()) == ["keep.txt"]
    assert (unrelated / "keep.txt").read_text() == "unrelated work"
    root = tmp_path / "job"
    job.prepare_job(root, **args)
    changed = replace(bundle.config, memory=Memory(device_gib=3, spill_gib=1))
    with pytest.raises(ValueError, match="different packages or setup"):
        job.prepare_job(root, **{**args, "config": changed})
    with pytest.raises(ValueError, match="different packages or setup"):
        job.prepare_job(root, **{**args, "packages": ["another-package"]})
    assert len(launches) == 1


def test_measurement_stops_before_planning_and_needs_no_initializer(
    prepared, tmp_path, monkeypatch
):
    bundle, _, _ = prepared
    config = replace(bundle.config, model=replace(bundle.config.model, initialize=None))
    launched = []

    def phases(package, root, selected, **options):
        assert job.PreparationJob.load(root / job.JOB_NAME).config == config
        launched.extend(options["phases"])

    monkeypatch.setattr(job, "run_phases", phases)
    root = job.measure_job(tmp_path / "measured", packages=(), config=config)
    assert root == tmp_path / "measured"
    assert launched == ["build", "profile", "communicate"]
    assert not (root / job.BUNDLE_NAME).exists()


def test_completed_bundle_validation_keeps_exclusive_job_ownership(
    prepared, tmp_path, monkeypatch
):
    bundle, _, _ = prepared
    original = job.TrainingBundle.load

    def load(path):
        with (
            pytest.raises(RuntimeError, match="already running"),
            job._job_lock(path.parent),
        ):
            pass
        return original(path)

    monkeypatch.setattr(job.TrainingBundle, "load", load)
    job.prepare_job(tmp_path / "job", packages=(), config=bundle.config)


def test_job_locks_survive_failure_and_prevent_concurrent_launch(prepared, tmp_path):
    bundle, _, launches = prepared
    root = tmp_path / "job"
    root.mkdir()
    args = {"packages": ["external-classifier-fixture"], "config": bundle.config}
    job.PreparationJob(tuple(args["packages"]), bundle.config).save(root / job.JOB_NAME)
    with job._job_lock(root):
        with pytest.raises(RuntimeError, match="already running"):
            job.prepare_job(root, **args)
        assert not launches
    job.prepare_job(root, **args)
    assert len(launches) == 1
    # Invalid completion is never treated as a cache miss or rerun.
    path = root / job.BUNDLE_NAME
    path.write_bytes(b"broken saved result")
    with pytest.raises(zipfile.BadZipFile):
        job.prepare_job(root, **args)
    assert len(launches) == 1
    assert path.read_bytes() == b"broken saved result"
    with job._job_lock(root):
        pass


def test_job_saved_record_rejects_unknown_fields_and_versions(prepared):
    bundle, _, _ = prepared
    value = job.PreparationJob(
        ("external-classifier-fixture",), bundle.config
    ).to_dict()
    for change in ({"version": True}, {"version": 2}, {"surprise": None}):
        with pytest.raises(ValueError, match="unsupported preparation job"):
            job.PreparationJob.from_dict({**value, **change})


def test_job_and_cpu_planning_entry_import_without_frameworks():
    code = """
import sys
for name in ('torch', 'triton', 'mlops', 'shadowspill_workloads'):
    sys.modules[name] = None
from shadowspill.training.preparation.phases import job, operations, plan
assert job.PreparationJob and operations.plan and plan.run
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr
