import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from shadowspill import Profiles, Program, apply_profiles, measure_program
from shadowspill.build import build_forward_backward, build_step
from tests.shadowspill.build.test_training import Network, objective


@pytest.mark.cuda
@pytest.mark.parametrize("with_optimizer", [False, True])
def test_saved_training_program_measurement_without_runtime_pools(
    tmp_path, with_optimizer
):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    with torch.device("meta"):
        model = Network()
    args = {
        "objective": objective,
        "example_args": (torch.empty(4, 8),),
        "example_kwargs": {"labels": torch.empty(4, 8)},
        "num_examples": 2,
    }
    if with_optimizer:
        program = build_step(
            model,
            optimizer=torch.optim.AdamW(model.parameters(), foreach=False),
            **args,
        )
    else:
        program = build_forward_backward(model, **args)
    program.save(tmp_path / "program.ssp")
    program = Program.load(tmp_path / "program.ssp")
    before = program.digest
    profiles = measure_program(program, warmup_seconds=0.001, measure_seconds=0.003)
    assert set(profiles.tasks) == {
        item.profile_key for item in program.task_definitions
    }
    assert program.digest == before
    assert torch.cuda.memory.get_allocator_backend() == "native"
    profiles.save(tmp_path / "profiles.ssp")
    profiles = Profiles.load(tmp_path / "profiles.ssp")
    assert profiles.fixed_device_bytes is not None
    assert profiles.metadata["external_headroom_bytes"] == 512 << 20
    assert profiles.fixed_device_bytes == (
        profiles.metadata["external_device_bytes"]
        + profiles.metadata["retained_device_bytes"]
        + profiles.metadata["external_headroom_bytes"]
    )
    profiled = apply_profiles(program, profiles)
    assert not profiled.estimated_memory
    for key, profile in profiles.tasks.items():
        if key != "invocation_start":
            assert profile.duration_seconds > 0
            assert profile.metadata["allocation_contract"]
            assert profile.metadata["compiled_layout"]
    subset = measure_program(
        program, stages=[0], warmup_seconds=0, measure_seconds=0.001
    )
    assert 1 < len(subset.tasks) < len(profiles.tasks)
    planned = subprocess.run(
        [
            sys.executable,
            "-S",
            str(Path(__file__).with_name("_plan_saved.py")),
            str(tmp_path),
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
    )
    assert planned.returncode == 0, planned.stdout + planned.stderr
    if not with_optimizer:
        environment = dict(os.environ)
        environment["TORCHINDUCTOR_CACHE_DIR"] = str(tmp_path / "fresh-inductor")
        environment["TRITON_CACHE_DIR"] = str(tmp_path / "fresh-triton")
        subprocess.run(
            [
                sys.executable,
                "-S",
                str(Path(__file__).with_name("_measure_saved.py")),
                str(tmp_path / "program.ssp"),
                str(tmp_path / "worker-profiles.ssp"),
            ],
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
    else:
        qualified = subprocess.run(
            [
                sys.executable,
                "-S",
                str(Path(__file__).with_name("_qualify_allocator.py")),
                str(tmp_path / "program.ssp"),
                str(tmp_path / "profiles.ssp"),
            ],
            capture_output=True,
            text=True,
        )
        assert qualified.returncode == 0, qualified.stdout + qualified.stderr
        assert "Qualified" in qualified.stdout
