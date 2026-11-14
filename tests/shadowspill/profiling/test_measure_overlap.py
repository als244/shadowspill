"""Bounded multi-GPU saved-code integration; no long model correctness campaign."""

import subprocess
import sys

import pytest

from shadowspill.orchestration.launch.processes import worker_command


@pytest.mark.cuda
@pytest.mark.slow
def test_saved_program_overlap_measurement_across_two_gpus(tmp_path):
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires two distinct CUDA devices")
    module = "tests.shadowspill.profiling._measure_overlap"
    for mode, count in (("build", 1), ("measure", 2)):
        process = subprocess.run(
            worker_command(module, (mode, tmp_path), ranks=count),
            capture_output=True,
            text=True,
            timeout=240,
        )
        assert process.returncode == 0, process.stdout + process.stderr
    # Saved enriched profiles and planning remain CPU-only.
    command = [
        sys.executable,
        "-c",
        "import sys; sys.modules['torch']=None; "
        "from shadowspill import Plan; p=Plan.load(sys.argv[1]); print(p.summary)",
        str(tmp_path / "plan.ssp"),
    ]
    process = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert process.returncode == 0, process.stdout + process.stderr
