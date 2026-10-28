"""Qwen's phase recipe preserves distributed inputs and bounded initialization."""

import json
import math
import os
import subprocess
import sys
import zipfile

import numpy as np
import pytest
import torch

from shadowspill.execution.tensors import TensorChunk
from workloads.models.qwen3moe.config import PRESETS
from workloads.models.shared.decoder import RotaryEmbedding
from workloads.recipes.text.data.calls import Inputs
from workloads.recipes.text.initialization import Initialize
from workloads.recipes.text.qwen3.common import batch_sizes


def test_replanning_preserves_prior_profile_headroom(tmp_path, monkeypatch):
    from workloads.recipes.text import train_qwen3_distributed as recipe

    calls = []
    monkeypatch.setattr(recipe, "run_phases", lambda *a, **k: calls.append((a, k)))
    data = tmp_path / "tokens"
    root = tmp_path / "run"
    recipe.main(["--phase", "build", "--data", str(data), "--run-dir", str(root)])
    path = root / "config.json"
    options = json.loads(path.read_text())
    assert options.pop("external_headroom_mib") == 768
    path.write_text(json.dumps(options))
    recipe.main(["--phase", "plan", "--run-dir", str(root), "--execution-gib", "4"])
    restored = json.loads(path.read_text())
    assert restored["external_headroom_mib"] == 512
    assert restored["execution_gib"] == 4
    assert [call[0][2] for call in calls] == ["build", "plan"]


def test_initializer_reconstructs_identical_shards_and_rotary_tables():
    config = PRESETS["tiny"]
    initialize = Initialize("qwen3moe", config, seed=41)
    initialize.block_elements = 257

    def fill(name, shape, dtype, width):
        tensor = torch.empty(math.prod(shape), dtype=dtype)
        for start in range(0, tensor.numel(), width):
            initialize(
                TensorChunk(name, shape, dtype, start, tensor.numel()),
                tensor[start : start + width],
            )
        return tensor.reshape(shape)

    rotary = RotaryEmbedding(config.rotary_width, base=config.rope_base, capacity=16)
    for name, shape, dtype in (
        ("blocks.0.moe.w13", (8, 64, 128), torch.bfloat16),
        ("blocks.0.moe.router.weight", (8, 128), torch.bfloat16),
        ("blocks.0.mixer.q_norm.weight", (32,), torch.bfloat16),
        ("rotary.cosine", (16, 32), torch.float32),
        ("rotary.sine", (16, 32), torch.float32),
    ):
        whole = fill(name, shape, dtype, math.prod(shape))
        # Each call is a disjoint part of one global tensor, as for DP shards.
        pieces = fill(name, shape, dtype, 113)
        torch.testing.assert_close(whole, pieces, rtol=0, atol=0)
        if name.startswith("rotary."):
            torch.testing.assert_close(whole, getattr(rotary, name.split(".")[1]))
        elif name.endswith("norm.weight"):
            assert torch.equal(whole, torch.ones_like(whole))
        else:
            assert abs(whole.float().std().item() - config.initializer_range) < 0.002


def test_rank_local_calls_normalize_by_global_tokens_and_restore_cursor(tmp_path):
    values = np.arange(4097, dtype=np.uint32)
    values.tofile(tmp_path / "train.bin")
    (tmp_path / "meta.json").write_text(json.dumps({"vocab_size": 8192}))
    options = {
        "data": str(tmp_path),
        "sequence_length": 16,
        "sequences_per_step": 4,
        "sequences_per_example": 2,
    }
    first = Inputs(options, rank=0, world_size=2)
    second = Inputs(options, rank=1, world_size=2)
    for rank, source in enumerate((first, second)):
        examples = next(source)
        tokens = torch.cat([args[0].flatten() for args in examples["args"]])
        assert torch.equal(tokens, torch.arange(rank * 64, (rank + 1) * 64))
        assert all(kwargs["normalizer"] == 1 / 128 for kwargs in examples["kwargs"])
        restored = Inputs(options, rank=rank, world_size=2)
        restored.load_state_dict(source.state_dict())
        torch.testing.assert_close(next(source), next(restored))
    assert batch_sizes({"sequences_per_step": 64, "max_sequences_per_example": 32}) == (
        1,
        2,
        4,
        8,
        16,
        32,
    )


@pytest.mark.cuda
@pytest.mark.parametrize("ranks", [1, 2])
@pytest.mark.slow
def test_qwen_recipe_builds_profiles_replans_and_executes_saved_plan(tmp_path, ranks):
    if torch.cuda.device_count() < ranks:
        pytest.skip(f"requires {ranks} distinct GPUs")
    pytest.importorskip("mlops")
    data, root = tmp_path / "data", tmp_path / "run"
    data.mkdir()
    (np.arange(4096, dtype=np.uint32) % 32).tofile(data / "train.bin")
    (data / "meta.json").write_text(json.dumps({"vocab_size": 1024}))
    command = [
        sys.executable,
        *(["-S"] if sys.flags.no_site else []),
        "-m",
        "workloads.recipes.text.train_qwen3_distributed",
        "--data",
        str(data),
        "--run-dir",
        str(root),
        "--model",
        "qwen3moe-tiny",
        "--ranks",
        str(ranks),
        "--sequence-length",
        "16",
        "--sequences-per-step",
        "2",
        "--max-sequences-per-example",
        "2",
        "--steps",
        "3",
        "--learning-rate",
        "0.001",
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
        timeout=900,
    )
    logs = "\n".join(path.read_text() for path in (root / "logs").glob("*.log"))
    assert child.returncode == 0, child.stdout + child.stderr + logs
    rows = json.loads((root / "search.json").read_text())
    assert len(rows) == 3 and all("duration_seconds" in row for row in rows)
    assert (root / "report/index.html").is_file()
    for rank in range(ranks):
        directory = root / f"training/rank-{rank:05d}"
        metrics = [
            json.loads(line)
            for line in (directory / "metrics.jsonl").read_text().splitlines()
        ]
        assert [row["step"] for row in metrics] == [1, 2, 3]
        assert all(math.isfinite(row["train/loss"]) for row in metrics)
        assert metrics[-1]["train/loss"] < metrics[0]["train/loss"]
        with zipfile.ZipFile(directory / "checkpoints/step_00000003.ssp") as archive:
            manifest = json.loads(archive.read("manifest.json"))
        assert any(item["dtype"] == "torch.float32" for item in manifest["tensors"])
    # Replan the actual saved measurements with CUDA hidden; even Torch imports
    # are forbidden here, not just CUDA execution.
    child = subprocess.run(
        [
            sys.executable,
            *(["-S"] if sys.flags.no_site else []),
            "-c",
            "import sys; sys.modules['torch'] = None; "
            "from workloads.recipes.text.qwen3.plan import run; run(sys.argv[1])",
            str(root),
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert child.returncode == 0, child.stdout + child.stderr
