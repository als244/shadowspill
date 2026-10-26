"""Real saved-Plan replay, semantic ranges and hardware metric validation."""

import json
import os
import shutil
import subprocess
import sys

import pytest

from benchmarking.quickstart import run


@pytest.mark.cuda
def test_saved_quickstart_produces_validated_nsys_capture(tmp_path):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    nsys = shutil.which("nsys")
    if nsys is None:
        pytest.skip("requires Nsight Systems and device-metric permission")
    root, captures = tmp_path / "quickstart", tmp_path / "nsys"
    run(
        "workloads.recipes.regression.benchmark:experiment",
        factory_args={"rows": 8, "width": 8, "outputs": 8},
        output_dir=root,
        search_budget_gib=[2],
        spill_gib=0.125,
        steps=3,
        warmup_seconds=0.001,
        measure_seconds=0.002,
        plots=False,
    )
    originals = {path: path.read_bytes() for path in root.glob("plans/*.ssp")}
    command = [
        sys.executable,
        "-S",
        "-m",
        "tools.pressurefit_trace.profile_winners",
        str(root),
        "--output",
        str(captures),
        "--nsys",
        nsys,
        "--warmup-steps",
        "1",
        "--capture-steps",
        "3",
    ]
    with (tmp_path / "capture.log").open("w") as log:
        result = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, env=os.environ, timeout=180
        )
    assert result.returncode == 0, (tmp_path / "capture.log").read_text()
    assert all(path.read_bytes() == value for path, value in originals.items())
    case = captures / "budget_2GiB"
    validated = json.loads((case / "validation.json").read_text())
    assert validated["status"] == "validated" and not validated["performance_result"]
    assert len(validated["step_ranges"]) == 3
    contract = json.loads(
        (case / "attempt-001/execution/rank-00000/exact-plan.json").read_text()
    )
    assert contract["plan_digest"] == validated["inputs"]["plan_digest"]
    reused = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert reused.returncode == 0, reused.stdout + reused.stderr
    assert "verified existing" in reused.stdout
    assert not (case / "attempt-002").exists()
