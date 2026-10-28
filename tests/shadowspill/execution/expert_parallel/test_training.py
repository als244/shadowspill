"""Two-H100 acceptance for saved provider resources and model-owned collectives."""

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.cuda
@pytest.mark.parametrize("resolution", [0, 1], ids=["save", "recompute"])
def test_saved_ep_training_and_checkpoint(tmp_path, resolution):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two distinct GPUs")
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("Quack EP currently requires SM90")
    script = (
        Path(__file__).parents[3] / "integration/pytorch/expert_parallel_training.py"
    )
    command = [
        sys.executable,
        *(["-S"] if sys.flags.no_site else []),
        str(script),
        "--resolution",
        str(resolution),
        "--out",
        str(tmp_path),
    ]
    with (
        (tmp_path / "qualification.log").open("w") as log,
        subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "OMP_NUM_THREADS": "1"},
        ) as child,
    ):
        try:
            status = child.wait(timeout=900)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            pytest.fail(f"EP qualification timed out; see {log.name}")
    assert status == 0, (tmp_path / "qualification.log").read_text()
    assert all((tmp_path / f"result-{rank}.json").exists() for rank in range(2))
