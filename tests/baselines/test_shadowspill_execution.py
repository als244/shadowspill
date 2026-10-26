"""Real catalog updates and oracle checks through the baseline's phase workers."""

import json
import os
import subprocess
import sys
from dataclasses import replace

import pytest

from baselines.common.launch import resolve, trainer_environment
from baselines.engines.shadowspill.settings import Settings
from benchmarking.quickstart.phases.common import write


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["full", "lora"])
@pytest.mark.parametrize("world", [1, 2])
def test_saved_baseline_matches_reference_in_fresh_worker(tmp_path, mode, world):
    import torch

    if torch.cuda.device_count() < world:
        pytest.skip(f"requires {world} CUDA devices")
    settings = Settings(
        model="llama3-tiny",
        mode=mode,
        lora={} if mode == "full" else {"rank": 32},
        model_config={"n_layers": 1},
        sequence_length=32,
        tokens_per_step=128 * world,
        tokens_per_microbatch=64,
        execution_gib=3,
        spill_gib=0.25,
        steps=2,
        warmup_steps=1,
        cpu_threads=2,
        verify_reference=True,
        fuse_narrow_gradient_accumulation=False,
        orderings="depth-first",
        resolution_options="0,1",
    )
    case = resolve("shadowspill", settings, world)
    write(tmp_path / "case.json", case)
    # The controller must tolerate a caller that already initialized CUDA.
    before = torch.ones(1, device="cuda")
    log = tmp_path / "console.log"
    with log.open("w") as output:
        child = subprocess.run(
            [
                sys.executable,
                "-S",
                "-m",
                "baselines.common.launch",
                "--engine",
                "shadowspill",
                "--gpus",
                str(world),
                "--python",
                sys.executable,
                "--output",
                str(tmp_path),
                *settings.arguments(),
            ],
            env=trainer_environment(case["environment"], os.environ),
            stdout=output,
            stderr=subprocess.STDOUT,
            timeout=900,
        )
    assert child.returncode == 0, log.read_text()[-16000:]
    assert before.item() == 1 and torch.cuda.get_allocator_backend() == "native"
    for rank in range(world):
        result = json.loads((tmp_path / f"rank-{rank:05d}/result.json").read_text())
        assert len(result["reference_checks"]) == 3
        assert result["simulation_comparison"]["median_measured_step_seconds"] > 0
    assert (tmp_path / "phases/planning/sweep.ssp").is_file()
    assert (tmp_path / "phases/report/index.html").is_file()
    assert (tmp_path / "row.json").is_file()
    if mode == "full" and world == 1:
        # Adding a budget reuses profiles and preserves the original executable.
        artifacts = {
            path: path.read_bytes()
            for folder in ("programs", "profiles", "profiled")
            for path in (tmp_path / "phases" / folder).glob("*.ssp")
        }
        saved = (tmp_path / "plan.ssp").read_bytes()
        later = tmp_path / "another-budget"
        settings = replace(settings, execution_gib=4)
        child = subprocess.run(
            [
                sys.executable,
                "-S",
                "-m",
                "baselines.common.launch",
                "--engine",
                "shadowspill",
                "--gpus",
                "1",
                "--python",
                sys.executable,
                "--output",
                str(later),
                "--prepared",
                str(tmp_path / "phases"),
                *settings.arguments(),
            ],
            env=trainer_environment(case["environment"], os.environ),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert child.returncode == 0, child.stdout + child.stderr
        assert "plan: starting" in child.stdout
        assert "build: starting" not in child.stdout
        assert "profile: starting" not in child.stdout
        assert "communicate: starting" not in child.stdout
        assert all(path.read_bytes() == value for path, value in artifacts.items())
        assert (tmp_path / "plan.ssp").read_bytes() == saved
        assert (later / "plan.ssp").is_file()
        reused = json.loads((later / "case.json").read_text())["phase_execution"]
        assert reused["reused_computation"]
