"""The non-text distributed recipe uses the public phase boundaries."""

import json
import math
import os
import subprocess
import sys
import zipfile

import pytest
import torch

from shadowspill.execution.tensors import TensorChunk
from workloads.recipes.regression.data import Inputs
from workloads.recipes.regression.initialize import fill_tensor


def test_regression_input_cursor_preserves_rank_local_examples():
    first, other = Inputs(0), Inputs(1)
    assert not torch.equal(first.example()["args"][0][0], other.example()["args"][0][0])
    next(first)
    state = first.state_dict()
    resumed = Inputs(0)
    resumed.load_state_dict(state)
    torch.testing.assert_close(next(first), next(resumed))
    with pytest.raises(ValueError, match="another rank"):
        other.load_state_dict(state)


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


def test_recipe_planning_entrypoint_does_not_import_torch(tmp_path):
    script = """
import importlib.abc, sys
class NoGPU(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('shadowspill.runtime'):
            raise AssertionError(f'CPU planning imported {fullname}')
sys.meta_path.insert(0, NoGPU())
import workloads.recipes.regression.plan
"""
    child = subprocess.run(
        [sys.executable, *(["-S"] if sys.flags.no_site else []), "-c", script],
        capture_output=True,
        text=True,
    )
    assert child.returncode == 0, child.stdout + child.stderr


@pytest.mark.cuda
def test_distributed_regression_runs_through_saved_artifacts(tmp_path):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two distinct GPUs")
    pytest.importorskip("mlops")
    root = tmp_path / "regression"
    command = [
        sys.executable,
        *(["-S"] if sys.flags.no_site else []),
        "-m",
        "workloads.recipes.distributed_training",
        "--run-dir",
        str(root),
        "--ranks",
        "2",
        "--steps",
        "3",
        "--execution-gib",
        "4",
        "--spill-gib",
        "0.125",
        "--warmup-seconds",
        "0.001",
        "--measure-seconds",
        "0.002",
        "--no-plots",
    ]
    child = subprocess.run(
        command,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=600,
    )
    logs = "\n".join(path.read_text() for path in (root / "logs").glob("*.log"))
    assert child.returncode == 0, child.stdout + child.stderr + logs
    assert (root / "report/index.html").is_file()
    for artifact in ("program", "profiles", "profiled", "communication", "plan"):
        assert (root / f"{artifact}.ssp").is_file()
    counters = []
    for rank in range(2):
        directory = root / f"training/rank-{rank:05d}"
        records = [
            json.loads(row)
            for row in (directory / "metrics.jsonl").read_text().splitlines()
        ]
        assert [row["step"] for row in records] == [1, 2, 3]
        assert all(math.isfinite(row["train/loss"]) for row in records)
        checkpoint = directory / "checkpoints/step_00000003.ssp"
        with zipfile.ZipFile(checkpoint) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            for tensor in manifest["tensors"]:
                if tensor["name"] == "updates" and tensor["bytes"]:
                    counters.append(
                        int.from_bytes(archive.read(tensor["member"]), "little")
                    )
    # The counter is logically one INT64: padding occupies the other rank's
    # physical shard, but only the actual counter belongs in the checkpoint.
    assert counters == [3]
