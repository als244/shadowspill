"""Real CUDA validates native task waits independently of NCCL availability."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.cuda
@pytest.mark.parametrize("kind", ["TASK_START", "TASK_ALLOCATION", "INTRA_TASK"])
def test_cross_stream_fixed_reuse(kind):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    result = subprocess.run(
        [
            sys.executable,
            *(["-S"] if sys.flags.no_site else []),
            str(Path(__file__).with_name("_task_completion.py")),
            kind,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "task completion passed" in result.stdout
