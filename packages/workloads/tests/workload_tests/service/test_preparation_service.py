"""Real target measurements exported over the public API to CPU planning."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.cuda
@pytest.mark.slow
@pytest.mark.parametrize("reuse", [False, True, "local"])
def test_public_measurement_exports_all_cases_to_cpu_planning(tmp_path, reuse):
    import torch

    from workload_tests.service.test_regression_service import initialized

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    (tmp_path / "setup.json").write_text(
        json.dumps(
            {
                "client_script": "_preparation_service_client.py",
                "model_description": "Regression measured at sizes 4 and 8",
                "oracle_description": "Original phase hashes and actual CPU replanning",
                "reuse": reuse,
                "guard_measurement_reuse": reuse,
                "local_planning": reuse == "local",
                "planner_python": os.environ.get(
                    "SHADOWSPILL_DEMO_PLANNER_PYTHON", sys.executable
                ),
                "initial": {
                    name: value.detach().tolist()
                    for name, value in initialized().named_parameters()
                },
            }
        )
    )
    server = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parent / "helpers/_catalog_service_server.py"),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=720,
    )
    (tmp_path / "server.stdout").write_text(server.stdout)
    (tmp_path / "server.stderr").write_text(server.stderr)
    assert server.returncode == 0, server.stdout + server.stderr
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert (
        receipt["all_server_claims_released"] and receipt["all_client_claims_released"]
    )
    assert receipt["captures"] >= 2
    if reuse:
        assert receipt["reused_training"] and len(receipt["whole_split_oracles"]) == 2
    if reuse == "local":
        assert receipt["local_planning"]["framework_free"]
        assert receipt["native_saved_plan_resume"]
    destination = tmp_path / "measurements.zip"
    if host := os.environ.get("SHADOWSPILL_DEMO_CLIENT_HOST"):
        copied = subprocess.run(
            ["scp", f"{host}:{receipt['archive']}", str(destination)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert copied.returncode == 0, copied.stderr
    else:
        shutil.copyfile(receipt["archive"], destination)
    with destination.open("rb") as stream:
        assert (
            hashlib.file_digest(stream, "sha256").hexdigest()
            == receipt["artifact"]["sha256"]
        )
    planned = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parent / "helpers/_plan_measurement_export.py"),
            str(tmp_path),
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=120,
    )
    (tmp_path / "planning.stdout").write_text(planned.stdout)
    (tmp_path / "planning.stderr").write_text(planned.stderr)
    assert planned.returncode == 0, planned.stdout + planned.stderr
    assert json.loads(planned.stdout)["planned_captures"] == receipt["captures"]
