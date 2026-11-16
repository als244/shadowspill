"""Only exact compatible measurements enter a new preparation job."""

import hashlib
import io
import json
import zipfile
from dataclasses import replace

import pytest

from engine_tests.training.test_measurement_export import measured as measured
from engine_tests.training.test_setup_resolution import installed
from shadowspill.training.preparation import job
from shadowspill.training.preparation.layouts import prepare_layouts
from shadowspill.training.preparation.measurement_export import export_measurements
from shadowspill.training.preparation.measurement_reuse import (
    REUSE_NAME,
    require_staged_reuse,
    stage_measurements,
)
from shadowspill.training.preparation.setup import resolve_training
from shadowspill.training.preparation.templates import capture_inventory
from shadowspill_client import Memory
from shadowspill_client.protocol.preparation import PREPARATION_MEDIA_TYPE
from shadowspill_client.protocol.requests import ArtifactRef

CONTEXT = {"server_incarnation": "current", "devices": ["gpu0"]}
REUSE = {"measurements": {"session_id": "source", "run_id": "measurement"}}


def reference(body):
    return ArtifactRef(
        "measurements",
        hashlib.sha256(body).hexdigest(),
        len(body),
        PREPARATION_MEDIA_TYPE,
    )


@pytest.fixture
def exported(measured, tmp_path):
    root, inv = measured
    path = export_measurements(
        root,
        tmp_path / "original.zip",
        context=CONTEXT,
        max_disk_bytes=8 << 20,
        max_files=128,
    )
    body = path.read_bytes()
    return inv, body, reference(body)


def target(inv, path, *, change=lambda config: config):
    config = change(inv.layouts.setup.config)
    config = replace(config, preparation=replace(config.preparation, reuse=REUSE))
    inv = capture_inventory(
        prepare_layouts(resolve_training(installed(), config=config))
    )
    path.mkdir()
    job.PreparationJob((), config).save(path / job.JOB_NAME)
    return inv


def stage(exported, root, inv, **overrides):
    _, body, ref = exported
    return stage_measurements(
        io.BytesIO(body),
        ref,
        root,
        inv,
        **{
            "context": CONTEXT,
            "max_disk_bytes": 8 << 20,
            "max_files": 128,
            **overrides,
        },
    )


def test_budget_changes_reuse_all_measured_cases_with_original_bytes(
    exported, tmp_path
):
    original, body, ref = exported
    root = tmp_path / "target"
    inv = target(original, root, change=lambda c: replace(c, memory=Memory(3, 2)))
    assert inv.templates.keys() == original.templates.keys()
    receipt = stage(exported, root, inv)
    assert receipt["calibration_reused"] and receipt["source"] == ref.to_dict()
    assert len(receipt["files"]) == 4 * len(inv.templates)
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        for name in receipt["files"]:
            assert (root / name).read_bytes() == archive.read(name)
    require_staged_reuse(root, inv.layouts.setup.config)
    _, _, restored = job.load_job(root)
    assert restored.templates.keys() == inv.templates.keys()
    # A chained export preserves the exact reuse provenance and original files.
    export_measurements(
        root, tmp_path / "reexport.zip", max_disk_bytes=8 << 20, max_files=128
    )
    with zipfile.ZipFile(tmp_path / "reexport.zip") as archive:
        assert json.loads(archive.read(REUSE_NAME)) == receipt


@pytest.mark.parametrize("change", ["profiling", "placement", "subset"])
def test_changed_policy_or_assignment_only_reuses_compatible_phases(
    exported, tmp_path, change
):
    original, _, _ = exported
    root = tmp_path / "target"

    def changed(config):
        prep = config.preparation
        if change == "profiling":
            prep = replace(
                prep, profiling=replace(prep.profiling, measurement_seconds=0.5)
            )
        if change == "subset":
            prep = replace(prep, cases=prep.cases[-1:])
        return replace(config, preparation=prep)

    inv = target(original, root, change=changed)
    receipt = stage(
        exported,
        root,
        inv,
        context={**CONTEXT, "devices": ["gpu1"]} if change == "placement" else CONTEXT,
    )
    assert not receipt["calibration_reused"]
    assert {name.split("/")[0] for name in receipt["files"]} == (
        {"programs"} if change == "profiling" else {"programs", "profiled"}
    )
    assert {
        name.split("/")[1][:-4] for name in receipt["files"]
    } == inv.templates.keys()
    assert not (root / "calibrated").exists()


@pytest.mark.parametrize(
    "damage", ["checksum", "member", "file_hash", "duplicate", "quota"]
)
def test_invalid_or_unbounded_exports_cannot_complete_staging(
    exported, tmp_path, damage
):
    original, body, ref = exported
    root = tmp_path / "target"
    inv = target(original, root)
    if damage in ("member", "file_hash", "duplicate"):
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            files = {name: archive.read(name) for name in archive.namelist()}
        manifest = json.loads(files["manifest.json"])
        key = next(iter(manifest["captures"]))
        if damage == "member":
            manifest["captures"][key]["files"]["programs"] = "../unrelated"
        if damage == "file_hash":
            manifest["files"][f"programs/{key}.ssp"]["sha256"] = "0" * 64
        files["manifest.json"] = json.dumps(manifest).encode()
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            for name, value in files.items():
                archive.writestr(name, value)
            if damage == "duplicate":
                with pytest.warns(UserWarning, match="Duplicate"):
                    archive.writestr("manifest.json", files["manifest.json"])
        body = stream.getvalue()
        ref = reference(body)
    if damage == "checksum":
        ref = replace(ref, sha256="0" * 64)
    with pytest.raises(ValueError):
        stage(
            (original, body, ref),
            root,
            inv,
            max_disk_bytes=1 if damage == "quota" else 8 << 20,
        )
    assert not (root / REUSE_NAME).exists()
    with pytest.raises(ValueError, match="staged by service"):
        job.load_job(root)
    assert not (tmp_path / "unrelated").exists()


def test_reference_alone_never_launches_or_changes_unrelated_directory(
    exported, tmp_path
):
    original, _, _ = exported
    config = original.layouts.setup.config
    config = replace(config, preparation=replace(config.preparation, reuse=REUSE))
    root = tmp_path / "missing"
    with pytest.raises(ValueError, match="staged by service"):
        job.measure_job(root, packages=(), config=config)
    assert not root.exists()
    root.mkdir()
    (root / "keep.txt").write_text("unrelated")
    with pytest.raises(ValueError, match="fresh owned"):
        stage(exported, root, original)
    assert (root / "keep.txt").read_text() == "unrelated"
