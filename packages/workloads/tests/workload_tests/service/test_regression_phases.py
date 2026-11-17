"""The public regression application, bounded initialization and exact resume."""

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from shadowspill.execution.tensors import TensorChunk
from shadowspill_workloads.models.regression.initialization import fill_tensor
from workload_tests.service.test_regression_service import initialized


def test_regression_initialization_agrees_across_shards_and_chunk_sizes():
    for name, shape, dtype in (
        ("network.0.weight", (32, 16), torch.float32),
        ("network.2.bias", (8,), torch.float32),
        ("updates", (), torch.int64),
    ):
        size = math.prod(shape)
        expected = torch.empty(size, dtype=dtype)
        fill_tensor(TensorChunk(name, shape, dtype, 0, size), expected)
        actual = torch.empty_like(expected)
        for start in range(0, size, 7):
            chunk = actual[start : start + 7]
            fill_tensor(TensorChunk(name, shape, dtype, start, size), chunk)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.cuda
@pytest.mark.slow
@pytest.mark.parametrize("ranks", [1, 2])
@pytest.mark.parametrize("pieces", [1, 3])
def test_regression_client_example_matches_oracle_and_resumes(tmp_path, ranks, pieces):
    if torch.cuda.device_count() < ranks:
        pytest.skip(f"requires {ranks} distinct GPUs")
    model = initialized()
    (tmp_path / "setup.json").write_text(
        json.dumps(
            {
                "client_script": "_regression_example_client.py",
                "ranks": ranks,
                "pieces": pieces,
                "initial": {
                    name: value.detach().tolist()
                    for name, value in model.named_parameters()
                },
                "model_description": f"Regression application, {ranks} DP workers",
                "oracle_description": (
                    "independent NumPy SiLU/AdamW loss and parameter deltas; "
                    "exact native resume"
                ),
            }
        )
    )
    child = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parent / "helpers/_catalog_service_server.py"),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )
    (tmp_path / "server.stdout").write_text(child.stdout)
    (tmp_path / "server.stderr").write_text(child.stderr)
    assert child.returncode == 0, child.stdout + child.stderr
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert receipt["actual_example"] and receipt["exact_checkpoint_continuation"]
    assert (
        receipt["independent_loss_and_weights"]
        and receipt["all_server_claims_released"]
    )
    assert receipt["ranks"] == ranks and receipt["pieces"] == pieces
