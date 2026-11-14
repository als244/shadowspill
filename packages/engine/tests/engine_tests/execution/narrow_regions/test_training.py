"""Actual FP16 storage across DP, chunking, retention and fresh-process resume."""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from engine_tests.execution.narrow_regions.fixture import CASES


def run_phase(directory, phase, *, distributed=False, cpu=False):
    module = (
        "fixture"
        if phase in {"prepare", "preflight"}
        else ("reference" if phase in {"reference", "reference_cpu"} else "phases")
    )
    command = [sys.executable, *(["-S"] if sys.flags.no_site else [])]
    if distributed:
        command += [
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            "--start-method=spawn",
            "--no-python",
            sys.executable,
            *(["-S"] if sys.flags.no_site else []),
        ]
    command += [str(Path(__file__).with_name(module + ".py")), str(directory)]
    if phase == "reference_cpu":
        command.append("--cpu-control")
    elif phase != "reference":
        command.append(phase)
    environment = {**os.environ, "OMP_NUM_THREADS": "1"}
    if cpu:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    with (directory / f"{phase}.log").open("w") as log:
        child = subprocess.Popen(
            command,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            status = child.wait(timeout=180)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            pytest.fail(f"{phase} timed out; see {log.name}")
    assert status == 0, (directory / f"{phase}.log").read_text()


@pytest.mark.cuda
@pytest.mark.parametrize("name", CASES)
def test_actual_fp16_regions_and_dp(tmp_path, name):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two distinct GPUs")
    (tmp_path / "case.json").write_text(json.dumps(CASES[name]) + "\n")
    run_phase(tmp_path, "prepare", cpu=True)
    run_phase(tmp_path, "preflight", cpu=True)
    run_phase(tmp_path, "reference")
    run_phase(tmp_path, "build")
    run_phase(tmp_path, "profile")
    run_phase(tmp_path, "communicate", distributed=True)
    run_phase(tmp_path, "plan", cpu=True)
    run_phase(tmp_path, "execute", distributed=True)
    run_phase(tmp_path, "resume", distributed=True)
    run_phase(tmp_path, "inspect", cpu=True)
    result = json.loads((tmp_path / "numerical-inspection.json").read_text())
    assert result["status"] == "PASS"
    assert result["fresh_resume"] == "BITWISE"
    assert result["physical_observations"] == 10
