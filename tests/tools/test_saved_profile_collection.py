"""Export, search and trace saved profiles without importing a model or Torch."""

import json
import os
import subprocess
import sys
from dataclasses import replace

import pytest

from benchmarking.quickstart.phases.common import read, write
from shadowspill import apply_profiles
from shadowspill.task.allocations import (
    TaskAllocationContract,
    TaskAllocationEvent,
    TaskAllocationOperation,
)
from shadowspill.task.layout import reconcile_compiled_task_layout
from shadowspill.task.profiles import TaskMeasurement
from shadowspill.task.storage import TaskStorageContract
from tests.shadowspill.offline._example import communication, profiled
from tools.pressurefit_trace.__main__ import main


def measured():
    original = profiled()
    profiles = original.profiles
    storage = TaskStorageContract.build((), (), ())
    for key, record in profiles.tasks.items():
        size = record.workspace_bytes
        trace = tuple(
            TaskAllocationEvent(0, operation, size, size)
            for operation in (
                TaskAllocationOperation.ALLOCATE,
                TaskAllocationOperation.FREE,
            )
            if size
        )
        measurement = TaskMeasurement(
            1000, size, size, (), (1000,), "fixture", allocation_trace=trace
        )
        profiles.tasks[key] = replace(
            record,
            metadata={
                "duration_kind": "static",
                "memory_kind": "static",
                "allocation_trace": [event.to_dict() for event in trace],
                "allocation_contract": TaskAllocationContract.capture(
                    trace, storage
                ).to_dict(),
                "compiled_layout": reconcile_compiled_task_layout(
                    storage, measurement
                ).identity(),
                "output_input_bindings": [],
                "off_device_output_leaves": [],
            },
        )
    profiles.metadata["memory_kind"] = "static"
    program = replace(
        original.program,
        executables={key: b"not executed" for key in profiles.tasks},
        bindings={
            "manifests": {
                key: {"storage_contract": storage.to_dict(), "root_allocations": []}
                for key in profiles.tasks
            },
            "entrypoints": [
                {"task_id": key, "input_slots": [], "output_slots": []}
                for key in profiles.tasks
            ],
        },
    )
    return apply_profiles(program, profiles)


@pytest.fixture
def saved_profiles(tmp_path):
    source = tmp_path / "quickstart"
    first = measured()
    for name, value in (
        ("first", first),
        ("second", replace(first, fixed_device_bytes=80)),
    ):
        value.save(source / f"profiled/{name}.ssp")
    write(
        source / "profiled/index.json",
        {
            "candidates": [
                {"name": name, "file": f"{name}.ssp"} for name in ("first", "second")
            ]
            + [{"name": "failed", "failure": {"type": "DeviceOutOfMemoryError"}}]
        },
    )
    write(
        source / "request.json",
        {
            "factory": "must_not_import:factory",
            "search_budgets": [4096],
            "spill_bytes": 4096,
        },
    )
    communication().save(source / "communication.ssp")
    return source


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
                         'benchmarking.quickstart.runner')
        ):
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from tools.pressurefit_trace.__main__ import main
for argv in json.loads(sys.argv[1]):
    main(argv)
"""
    commands = [
        ["dump", str(saved_profiles), "--output", str(output)],
        ["plan", str(output), "--output", str(plans), "--workers", "1"],
        ["trace", str(plans), "--output", str(traces), "--programs", "all"],
        ["export", str(traces), "--output", str(exported)],
    ]
    child = subprocess.run(
        [sys.executable, "-S", "-c", script, json.dumps(commands)],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    assert all(path.read_bytes() == value for path, value in original.items())
    capture = read(output / "dump.json")
    assert capture["execution_budget_bytes"] == 4096 - 80
    assert len(capture["programs"]) == 2 and len(capture["infeasible_geometries"]) == 1
    assert capture["fixed_device_bytes_by_candidate"] == {"first": 20, "second": 80}
    semantics = read(output / capture["programs"][0]["semantics"])
    assert semantics["objects"]["weight"]["label"] == "linear.weight"
    assert semantics["tasks"]["update"]["module_paths"] == ["linear"]
    assert len(read(exported / "collection.json")["programs"]) == 2


def test_estimated_memory_is_not_promoted_to_allocation_evidence(
    saved_profiles, tmp_path
):
    profiled().save(saved_profiles / "profiled/first.ssp")
    with pytest.raises(SystemExit) as failure:
        main(["dump", str(saved_profiles), "--output", str(tmp_path / "bad")])
    assert failure.value.code == 2


def test_larger_dump_cap_reuses_profiles(saved_profiles, tmp_path):
    target = tmp_path / "larger"
    main(
        [
            "dump",
            str(saved_profiles),
            "--output",
            str(target),
            "--physical-gib",
            str(8192 / (1 << 30)),
        ]
    )
    capture = read(target / "dump.json")
    assert capture["physical_budget_bytes"] == 8192
    assert capture["execution_budget_bytes"] == 8192 - 80
