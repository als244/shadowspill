import json
import zipfile
from dataclasses import replace

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
