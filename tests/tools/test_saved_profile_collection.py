"""Export, search and trace saved profiles without importing a model or Torch."""

import json
import os
import subprocess
import sys
from dataclasses import replace

import pytest

from engine_tests.offline._example import communication, profiled
from tests.tools.planning._profiled import measured
from tools.pressurefit_trace.__main__ import main


@pytest.fixture
def saved_profiles(tmp_path):
    source = tmp_path / "profiles"
    first = measured()
    for name, value in (
        ("first", first),
        ("second", replace(first, fixed_device_bytes=80)),
    ):
        value.save(source / f"{name}.ssp")
    communication().save(source / "communication.ssp")
    return source


def read(path):
    return json.loads(path.read_text())


def dump_command(source, output, physical=4096):
    return [
        "dump",
        str(source / "first.ssp"),
        str(source / "second.ssp"),
        "--communication",
        str(source / "communication.ssp"),
        "--physical-gib",
        str(physical / (1 << 30)),
        "--spill-gib",
        str(4096 / (1 << 30)),
        "--output",
        str(output),
    ]


def test_saved_collection_is_cpu_only_and_preserves_evidence(saved_profiles, tmp_path):
    output, plans, traces, exported = (
        tmp_path / p for p in ("dump", "plans", "traces", "export")
    )
    original = {path: path.read_bytes() for path in saved_profiles.rglob("*.ssp")}
    script = """
import importlib.abc
import json
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'must_not_import'} or any(
            fullname == name or fullname.startswith(name + '.')
            for name in ('shadowspill.build', 'shadowspill.execution',
                         'shadowspill.training.preparation')
        ):
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from tools.pressurefit_trace.__main__ import main
for argv in json.loads(sys.argv[1]):
    main(argv)
"""
    commands = [
        dump_command(saved_profiles, output),
        ["plan", str(output), "--output", str(plans), "--workers", "1"],
        ["trace", str(plans), "--output", str(traces), "--programs", "all"],
        ["export", str(traces), "--output", str(exported)],
    ]
    child = subprocess.run(
        [
            sys.executable,
            *(["-S"] if sys.flags.no_site else []),
            "-c",
            script,
            json.dumps(commands),
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    assert all(path.read_bytes() == value for path, value in original.items())
    capture = read(output / "dump.json")
    assert capture["execution_budget_bytes"] == 4096 - 80
    assert len(capture["programs"]) == 2
    assert capture["infeasible_geometries"] == []
    assert capture["fixed_device_bytes_by_candidate"] == {"first": 20, "second": 80}
    semantics = read(output / capture["programs"][0]["semantics"])
    assert semantics["objects"]["weight"]["label"] == "linear.weight"
    assert semantics["tasks"]["update"]["module_paths"] == ["linear"]
    assert len(read(exported / "collection.json")["programs"]) == 2


def test_estimated_memory_is_not_promoted_to_allocation_evidence(
    saved_profiles, tmp_path
):
    profiled().save(saved_profiles / "first.ssp")
    with pytest.raises(SystemExit) as failure:
        main(dump_command(saved_profiles, tmp_path / "bad"))
    assert failure.value.code == 2


def test_larger_dump_cap_reuses_profiles(saved_profiles, tmp_path):
    target = tmp_path / "larger"
    main(dump_command(saved_profiles, target, physical=8192))
    capture = read(target / "dump.json")
    assert capture["physical_budget_bytes"] == 8192
    assert capture["execution_budget_bytes"] == 8192 - 80
