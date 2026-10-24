"""NCCL singleton test validates the memory path on single-GPU development hosts."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.cuda
def test_collective_allocation_observation(tmp_path):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(Path(__file__).with_name("_collective_memory.py")),
            (tmp_path / "rendezvous").as_uri(),
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "collective memory observed without execution pools" in result.stdout
