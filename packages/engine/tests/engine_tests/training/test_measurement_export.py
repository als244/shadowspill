"""Export structure/storage proofs; GPU tests separately produce real captures."""

import hashlib
import json
import subprocess
import sys
import zipfile

import pytest

from engine_tests.training.test_preparation_templates import inventory
from engine_tests.training.test_setup_resolution import installed
from shadowspill.training.preparation.artifacts.measurement_export import (
    export_measurements,
)
from shadowspill.training.preparation.phases import job


@pytest.fixture
def measured(tmp_path, monkeypatch):
    prepared = inventory()
    monkeypatch.setattr(job, "load_registry", lambda _: installed())
    root = tmp_path / "measurements"
    root.mkdir()
    config = prepared.layouts.setup.config
    job.PreparationJob((), config).save(root / job.JOB_NAME)
    for stage in ("programs", "profiled", "calibrated", "communication"):
        (root / stage).mkdir()
        for key in prepared.templates:
            # Producer contents are opaque to this streaming export layer.
            (root / stage / f"{key}.ssp").write_bytes(f"{stage}:{key}".encode())
    return root, prepared


def export(root, destination, *, max_disk_bytes=8 << 20, max_files=128):
    return export_measurements(
        root, destination, max_disk_bytes=max_disk_bytes, max_files=max_files
    )


def test_export_retains_every_case_and_original_phase_file(measured, tmp_path):
    root, prepared = measured
    before = {p: p.read_bytes() for p in root.rglob("*.ssp")}
    first = export(root, tmp_path / "first.zip")
    second = export(root, tmp_path / "second.zip")
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["format"] == "shadowspill.operator_preparation-export"
        assert manifest["configuration"] == prepared.layouts.setup.config.to_dict()
        assert manifest["captures"].keys() == prepared.templates.keys()
        assert all(row["required"] for row in manifest["captures"].values())
        assert not manifest["rejected"]
        assert set(archive.namelist()) == {"manifest.json", *manifest["files"]}
        for name, info in manifest["files"].items():
            body = archive.read(name)
            assert body == before[root / name]
            assert info == {
                "size_bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
            }
    assert all(p.read_bytes() == content for p, content in before.items())


@pytest.mark.parametrize("damage", ["missing", "extra", "link", "file_count"])
def test_incomplete_or_unsafe_exports_never_create_output(measured, tmp_path, damage):
    root, prepared = measured
    key = next(iter(prepared.templates))
    if damage == "missing":
        (root / "calibrated" / f"{key}.ssp").unlink()
    elif damage == "extra":
        (root / "programs" / ("f" * 64 + ".ssp")).write_bytes(b"unrequested")
    elif damage == "link":
        (root / "link").symlink_to(tmp_path / "unrelated")
    destination = tmp_path / "export.zip"
    with pytest.raises(ValueError):
        export(root, destination, max_files=2 if damage == "file_count" else 128)
    assert not destination.exists()


def test_export_growth_obeys_shared_disk_bound_and_preserves_unrelated_files(
    measured, tmp_path
):
    root, prepared = measured
    payload = root / "calibrated" / f"{next(iter(prepared.templates))}.ssp"
    payload.write_bytes(b"x" * (2 << 20))
    destination = tmp_path / "export.zip"
    with pytest.raises(ValueError, match="reserved disk envelope"):
        export(root, destination, max_disk_bytes=3 << 20)
    assert not destination.exists() and payload.stat().st_size == 2 << 20
    export(root, destination)
    original = destination.read_bytes()
    with pytest.raises(ValueError, match="new path"):
        export(root, destination)
    assert destination.read_bytes() == original
    with pytest.raises(ValueError, match="outside"):
        export(root, root / "nested.zip")


def test_export_and_cpu_manifest_need_no_tensor_framework(measured, tmp_path):
    root, _ = measured
    code = """
import sys
from pathlib import Path
for name in ('torch', 'triton', 'mlops', 'shadowspill_workloads'):
    sys.modules[name] = None
from shadowspill.training.preparation.phases import job
from shadowspill.training.preparation.artifacts.measurement_export import (
    export_measurements,
)
from engine_tests.training.test_setup_resolution import installed
job.load_registry = lambda packages: installed()
export_measurements(
    Path(sys.argv[1]), Path(sys.argv[2]), max_disk_bytes=8<<20, max_files=128,
)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(root), str(tmp_path / "cpu.zip")],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
