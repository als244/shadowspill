"""Exercise the collection CLI, resume, and CPU replanning on real task profiles."""

import json
import os
import subprocess
import sys

import pytest

from benchmarking.program_collection.artifacts import load_program
from benchmarking.program_collection.matrix import expand_program_requests
from benchmarking.program_collection.settings import (
    CollectionConfig,
    GeometryAxes,
    ModelSpec,
    ProfilingSpec,
)
from benchmarking.program_collection.state import CollectionPaths, load_case_status


@pytest.mark.cuda
def test_collection_uses_native_allocator_and_saved_phase_archive(tmp_path):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    config = CollectionConfig(
        name="tiny-phase-corpus",
        seed=17,
        expected_programs=1,
        case_timeout_seconds=180,
        max_attempts=1,
        geometry=GeometryAxes((64,), (32,), (2,)),
        models=(ModelSpec("llama-tiny", "llama3-tiny", "mlops"),),
        profiling=ProfilingSpec(warmup_seconds=0.001, measure_seconds=0.003),
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.to_dict()))
    output = tmp_path / "corpus"
    args = ["--config", str(path), "--output-dir", str(output)]
    guard = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = ("shadowspill.execution", "shadowspill.pytorch.runtime")
        if fullname.startswith(blocked):
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
"""
    worker = (
        guard + "\nfrom benchmarking.program_collection.worker import main\n"
        "raise SystemExit(main())"
    )
    controller = (
        guard
        + f"""
from benchmarking.program_collection import controller
original = controller._worker_command
def command(*args):
    values = original(*args)
    index = values.index("-m")
    return [*values[:index], "-c", {worker!r}, *values[index + 2:]]
controller._worker_command = command
from benchmarking.program_collection.cli import main
raise SystemExit(main())
"""
    )
    command = [sys.executable, "-S", "-c", controller, *args]
    child = subprocess.run(command, capture_output=True, text=True, timeout=240)
    assert child.returncode == 0, child.stdout + child.stderr
    (request,) = expand_program_requests(config)
    paths = CollectionPaths.initialize(output, config)
    status = load_case_status(paths, request)
    saved, program = load_program(output / status["artifact"]["directory"])
    assert program.metadata["allocator"] == "native"
    assert program.program.bindings["kind"] == "step"
    assert len(program.program.bindings["training_templates"]) == 2
    assert not program.estimated_memory
    assert program.program.executables
    payload = saved.program_path.read_bytes()
    resumed = subprocess.run(
        [*command, "--resume"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert "PROGRAM SKIP" in resumed.stdout
    assert saved.program_path.read_bytes() == payload

    planned = subprocess.run(
        [
            sys.executable,
            "-S",
            "-m",
            "tests.benchmarking._plan_corpus",
            str(tmp_path),
            str(saved.directory),
            str(3 << 30),
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert planned.returncode == 0, planned.stdout + planned.stderr
