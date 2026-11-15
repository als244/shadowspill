"""Opt-in remote inspection of genuine, previously qualified training storage."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(
    not os.environ.get("SHADOWSPILL_METRICS_HISTORY_ROOT"),
    reason="requires retained qualified GPU service storage and a prepared demo client",
)
@pytest.mark.parametrize(
    "case", ["test_public_uploaded_checkpoin0", "test_public_uploaded_checkpoin1"]
)
def test_real_full_and_frozen_training_metrics_survive_remote_restart(tmp_path, case):
    source = Path(os.environ["SHADOWSPILL_METRICS_HISTORY_ROOT"]) / case
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("_metrics_history_server.py")),
            str(source),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=390,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert receipt["all_claims_released"] and receipt["no_gpu_or_training_workers"]
    assert all(
        row["qualified_losses_match"] and row["jsonl_roundtrip"]
        for row in receipt["runs"]
    )
    if os.environ.get("SHADOWSPILL_METRICS_ANNOTATE") == "1":
        assert all(row["caller_metrics_verified"] for row in receipt["runs"])
