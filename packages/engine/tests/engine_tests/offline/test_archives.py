import hashlib
import json
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

from shadowspill import (
    MemoryBudgets,
    Plan,
    Profiles,
    plan_program,
    sweep_programs,
    write_report,
)

from ._example import communication, options, profiled, profiles, program


def test_immutable_program_reuses_identity_but_replacement_recomputes(monkeypatch):
    source = program()
    original = type(source)._to_dict
    calls = []

    def counted(value, **kwargs):
        calls.append(value)
        return original(value, **kwargs)

    monkeypatch.setattr(type(source), "_to_dict", counted)
    first = source.digest
    assert source.digest == first
    assert len(calls) == 1
    changed = replace(source, name=source.name + " changed")
    assert changed.digest != first
    assert len(calls) == 2
    assert source.digest == first


def test_phase_artifacts_are_self_contained_single_archives(tmp_path):
    plan = plan_program(
        profiled(),
        memory_budgets=MemoryBudgets(100, 64),
        communication_info=communication(),
        search_options=options(),
    )
    sweep = sweep_programs(
        {"update": profiled()},
        memory_budgets=[MemoryBudgets(100, 64)],
        communication_info=communication(),
        search_options=options(),
    )
    for index, artifact in enumerate(
        (program(), profiles(), profiled(), plan, sweep, communication())
    ):
        path = tmp_path / str(index)
        artifact.save(path)
        loaded = type(artifact).load(path)
        assert loaded.to_dict() == artifact.to_dict()
        with zipfile.ZipFile(path) as archive:
            assert sorted(archive.namelist()) == ["artifact.json", "manifest.json"]
    assert len(list(tmp_path.iterdir())) == 6


@pytest.mark.parametrize("limit", [0, 128, "almost"])
def test_bounded_archive_save_preserves_original_and_never_overwrites_allowance(
    tmp_path, monkeypatch, limit
):
    source = program()
    path = tmp_path / "program.ssp"
    source.save(path)
    original = path.read_bytes()
    limit = len(original) - 1 if limit == "almost" else limit
    removed = []
    unlink = Path.unlink

    def check_cleanup(file, *args, **kwargs):
        if file.name.startswith(".program.ssp."):
            removed.append(file.stat().st_size)
        return unlink(file, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", check_cleanup)
    with pytest.raises(ValueError, match="reserved disk envelope"):
        source.save(path, max_bytes=limit)
    assert path.read_bytes() == original
    assert removed and max(removed) <= limit
    assert list(tmp_path.iterdir()) == [path]
    source.save(path, max_bytes=len(original))
    assert type(source).load(path).to_dict() == source.to_dict()


def test_archive_rejects_wrong_kind_and_corrupt_payload(tmp_path):
    path = tmp_path / "program.ssp"
    program().save(path)
    with pytest.raises(ValueError, match="unsupported archive"):
        Profiles.load(path)
    with zipfile.ZipFile(path) as archive:
        manifest = archive.read("manifest.json")
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", manifest)
        archive.writestr("artifact.json", "{}")
    with pytest.raises(ValueError, match="checksum"):
        type(program()).load(path)


def test_nested_payloads_are_packed_once_and_checked(tmp_path):
    source = profiled()
    source = replace(
        source,
        program=replace(source.program, executables={"one": b"code", "two": b"code"}),
    )
    sweep = sweep_programs(
        {"a": source, "b": source},
        memory_budgets=[MemoryBudgets(100, 64)],
        communication_info=communication(),
        search_options=options(),
    )
    path = tmp_path / "sweep.ssp"
    sweep.save(path)
    restored = type(sweep).load(path)
    assert restored.to_dict() == sweep.to_dict()
    assert restored._archive_payloads() == sweep._archive_payloads()
    with zipfile.ZipFile(path) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    blobs = [name for name in contents if name.startswith("blobs/")]
    assert len(blobs) == 1
    contents[blobs[0]] = b"wrong"
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in contents.items():
            archive.writestr(name, value)
    with pytest.raises(ValueError, match="binary payload checksum"):
        type(sweep).load(path)


def test_report_from_restored_plan_has_numeric_rows_and_figures(tmp_path):
    plan = plan_program(
        profiled(),
        memory_budgets=MemoryBudgets(100, 64),
        communication_info=communication(2e-6),
        search_options=options(),
    )
    plan.save(tmp_path / "plan.ssp")
    restored = Plan.load(tmp_path / "plan.ssp")
    index = write_report(restored, tmp_path / "report")
    assert "linear.weight update" in index.read_text()
    assert "Simulation only" in index.read_text()
    assert json.loads((index.parent / "summary.json").read_text())["fetch_bytes"] == 64
    assert (index.parent / "timeline.svg").stat().st_size > 100
    assert (index.parent / "timeline.png").stat().st_size > 100
    report = (index.parent / "occupancy.html").read_text()
    payload = json.loads(
        report.split('type="application/json">', 1)[1].split("</script>", 1)[0]
    )
    assert payload["summary"]["seconds"] == restored.timeline.makespan_ns / 1e9
    assert payload["allocation"] == restored.summary
    assert payload["compute"] and payload["fetch"]
    assert {pool["name"] for pool in payload["pools"]} == {"spill"}
    assert payload["simulation_only_reasons"]  # No invented physical layout.


@pytest.mark.parametrize("limit", ["metadata", "payload"])
def test_archive_size_limits_precede_any_content_read(tmp_path, monkeypatch, limit):
    source = program()
    source = replace(source, executables={"one": b"x" * 4096})
    path = tmp_path / "bounded.ssp"
    source.save(path)
    with zipfile.ZipFile(path) as archive:
        metadata = sum(
            archive.getinfo(name).file_size
            for name in ("manifest.json", "artifact.json")
        )
    assert (
        type(source)
        .load(path, max_metadata_bytes=metadata, max_payload_bytes=4096)
        .to_dict()
        == source.to_dict()
    )

    def no_read(*args, **kwargs):
        pytest.fail("read content before checking byte limits")

    monkeypatch.setattr(zipfile.ZipFile, "read", no_read)
    with pytest.raises(ValueError, match=f"{limit} byte limit"):
        type(source).load(
            path, **{f"max_{limit}_bytes": 4095 if limit == "payload" else 1}
        )


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b'{"x":1,"x":2}', "duplicate archive JSON"),
        (b'{"x":NaN}', "nonfinite archive JSON"),
        (b"[]", "must contain an object"),
    ],
)
def test_archive_rejects_ambiguous_metadata_even_with_valid_checksum(
    tmp_path, data, message
):
    path = tmp_path / "invalid.ssp"
    source = program()
    source.save(path)
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    manifest["sha256"] = hashlib.sha256(data).hexdigest()
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("artifact.json", data)
    with pytest.raises(ValueError, match=message):
        type(source).load(path)
