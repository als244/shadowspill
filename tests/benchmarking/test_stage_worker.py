"""The diagnostic worker compiles saved pairs through the same path as build."""

import json
import subprocess
import sys

import pytest
import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode

from shadowspill.pytorch.capture.fake import fake_device_inputs, fake_device_model
from shadowspill.pytorch.capture.objective import capture_training_objective
from shadowspill.pytorch.graph_pairs import partition_training_capture
from shadowspill.pytorch.graph_pairs.serialization import CachedAotGraphPair


@pytest.mark.cuda
@pytest.mark.parametrize("variant", ["save", "recompute"])
def test_stage_worker_replays_saved_pair(tmp_path, variant):
    with FakeTensorMode(allow_non_fake_inputs=True) as mode:
        model = fake_device_model(nn.Sequential(nn.Linear(8, 8), nn.SiLU()), mode)
        inputs = fake_device_inputs((torch.empty(4, 8),), mode)
        capture = capture_training_objective(model, lambda m, x: m(x).sum(), inputs)
        stage = partition_training_capture(capture, partition="whole").stages[0]
    pair = next(v.pair for v in stage.graph_pairs.variants if v.option_id == variant)
    artifact = tmp_path / "pair.pt"
    torch.save(
        {
            "pair": CachedAotGraphPair.capture(pair),
            "policies": [
                {"kind": "normal", "source": str(i), "seed": 42 + i, "std": 0.1}
                for i in range(len(pair.forward.example_arguments))
            ],
            "variant": variant,
            "workload": {"sequence_length": 4, "batch_size": 1},
        },
        artifact,
    )
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-m",
            "benchmarking.stage_traces.worker",
            "--artifact",
            str(artifact),
            "--output",
            str(tmp_path),
            "--stage",
            "test",
            "--warmup",
            "1",
            "--iterations",
            "2",
        ],
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads((tmp_path / "result.json").read_text())
    assert record["finite_outputs_and_gradients"]
    assert len(record["samples"]) == 2
    assert all(s["forward_ms"] > 0 and s["backward_ms"] > 0 for s in record["samples"])
