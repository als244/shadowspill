"""Opt-in GPU checks of the public catalog runner, including evaluation/resume."""

import json
import math

import numpy as np
import pytest
import torch

from workloads.recipes import Model, train


@pytest.mark.cuda
@pytest.mark.slow
@pytest.mark.parametrize("name", ["llama3-tiny", "olmoe-tiny"])
def test_catalog_training_searches_and_resumes_in_fresh_workers(tmp_path, name):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    pytest.importorskip("mlops")
    data = tmp_path / "data"
    data.mkdir()
    # Repeated token sequences make a small, learnable next-token test corpus.
    tokens = np.arange(4096, dtype=np.uint32) % 32
    for split in ("train", "val"):
        tokens.tofile(data / f"{split}.bin")
    (data / "meta.json").write_text(json.dumps({"vocab_size": 32}))
    model = Model.from_config(name)
    root = train(
        model,
        {
            "run_dir": tmp_path / "run",
            "data": data,
            "sequence_length": 16,
            "sequences_per_step": 4,
            "microbatch_sizes": [1, 2],
            "steps": 3,
            "schedule_steps": 4,
            "learning_rate": 0.001,
            "execution_gib": 4,
            "spill_gib": 0.125,
            "warmup_seconds": 0.001,
            "measure_seconds": 0.002,
            "plots": False,
        },
    )
    rows = json.loads((root / "search.json").read_text())
    assert {row["sequences_per_example"] for row in rows} == {1, 2}
    assert all("duration_seconds" in row for row in rows)
    metrics = root / "training/metrics.jsonl"
    updates = [json.loads(line) for line in metrics.read_text().splitlines()]
    assert [row["step"] for row in updates] == [1, 2, 3]
    assert all(math.isfinite(row["train/loss"]) for row in updates)
    assert updates[-1]["train/loss"] < updates[0]["train/loss"]
    checks = root / "training/evaluation.jsonl"
    evaluations = [json.loads(line) for line in checks.read_text().splitlines()]
    assert [row["step"] for row in evaluations] == [2, 3]
    assert all(math.isfinite(row["validation_loss"]) for row in evaluations)
    checkpoint = root / "training/checkpoints/step_00000003.ssp"
    assert checkpoint.is_file()
    train(
        model,
        {"run_dir": root, "resume": checkpoint, "steps": 4},
        phase="execute",
    )
    updates = [json.loads(line) for line in metrics.read_text().splitlines()]
    assert [row["step"] for row in updates] == [1, 2, 3, 4]
    assert math.isfinite(updates[-1]["train/loss"])
    assert json.loads((root / "completed.json").read_text())["steps"] == 4
