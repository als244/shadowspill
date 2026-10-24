import subprocess
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.cuda
def test_communication_calibration_without_runtime(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(Path(__file__).with_name("_measure_communication.py")),
            str(tmp_path / "communication.ssp"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
