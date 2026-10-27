"""A real family checkpoint through bounded loading and the saved-Plan recipe."""

import json
import math
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from shadowspill.execution.tensors import TensorChunk
from tests.workloads.test_checkpoints import built, huggingface_state, metadata
from workloads.lora import configure_lora
from workloads.recipes.text.qwen35_lora.initialize import Initialize


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(123)
    model = built("qwen35moe-tiny", max_seq_len=32)
    root = tmp_path / "base"
    root.mkdir()
    save_file(
        huggingface_state("qwen35moe", model, model.config), root / "model.safetensors"
    )
    (root / "config.json").write_text(json.dumps(metadata("qwen35moe", model.config)))
    return root, model


def test_bounded_initializer_preserves_base_and_adapter_distributions(checkpoint):
    root, base = checkpoint
    expected = dict(base.named_parameters()) | dict(base.named_buffers())
    model = configure_lora(built("qwen35moe-tiny", max_seq_len=32), rank=32, alpha=32)
    options = {"checkpoint": root, "sequence_length": 32, "seed": 45}
    with Initialize(options) as initialize:
        initialize.block_elements = 257
        for name, tensor in (*model.named_parameters(), *model.named_buffers()):
            values = []
            for width in (tensor.numel(), 113):
                value = torch.empty(tensor.numel(), dtype=tensor.dtype)
                for start in range(0, value.numel(), width):
                    request = TensorChunk(
                        name,
                        tuple(tensor.shape),
                        tensor.dtype,
                        start,
                        math.prod(tensor.shape),
                    )
                    initialize(request, value[start : start + width])
                values.append(value.reshape(tensor.shape))
            torch.testing.assert_close(*values, rtol=0, atol=0)
            if name in expected:
                torch.testing.assert_close(values[0], expected[name], rtol=0, atol=0)
            elif name.endswith("_b"):
                assert values[0].count_nonzero() == 0
            else:
                assert values[0].count_nonzero() > 0
                if name.endswith(".lora_a"):
                    assert values[0].abs().max() <= tensor.shape[-1] ** -0.5 + 0.001
    assert initialize.checkpoint.reader._closed


def manifest(root, step):
    with zipfile.ZipFile(root / f"training/checkpoints/step_{step:08d}.ssp") as archive:
        return json.loads(archive.read("manifest.json"))


@pytest.mark.cuda
def test_qwen_recipe_cold_execution_and_partial_resume(tmp_path, checkpoint):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    base, _ = checkpoint
    root, data = tmp_path / "run", tmp_path / "data"
    data.mkdir()
    tokens = np.arange(1025, dtype=np.uint32) % 32
    tokens.tofile(data / "train.bin")
    ((tokens + 7) % 32).tofile(data / "val.bin")
    (data / "meta.json").write_text(json.dumps({"vocab_size": 1024}))
    cache = tmp_path / "preparation-cache"
    environment = {
        **os.environ,
        "TRITON_CACHE_DIR": str(cache / "triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"),
    }
    command = [
        sys.executable,
        "-S",
        "-m",
        "workloads.recipes.text.train_qwen35_lora",
        "--data",
        str(data),
        "--checkpoint",
        str(base),
        "--frozen-state-id",
        "tiny-qwen-base",
        "--run-dir",
        str(root),
        "--sequence-length",
        "16",
        "--sequences-per-step",
        "2",
        "--steps",
        "3",
        "--execution-gib",
        "3",
        "--spill-gib",
        "0.25",
        "--warmup-seconds",
        "0.001",
        "--measure-seconds",
        "0.002",
        "--no-plots",
    ]
    child = subprocess.run(
        command, env=environment, capture_output=True, text=True, timeout=900
    )
    logs = "\n".join(path.read_text() for path in (root / "logs").glob("*.log"))
    assert child.returncode == 0, child.stdout + child.stderr + logs
    from shadowspill import Program

    built = Program.load(root / "programs/batch-1/training.ssp")
    assert any(built.bindings["triton_choices"].values())
    assert any(
        item.compatibility_digest != item.profile_key
        for item in built.task_definitions
        if item.profile_key in built.bindings["triton_choices"]
    )
    search = json.loads((root / "search.json").read_text())
    assert {row["sequences_per_example"] for row in search} == {1, 2}
    assert all("duration_seconds" in row for row in search)
    expected = manifest(root, 3)
    assert "embed.weight" in expected["omitted_parameters"]
    assert all("lora" not in name for name in expected["omitted_parameters"])
    assert not list(root.rglob("*.cubin"))
    shutil.rmtree(cache)
    cold = tmp_path / "cold"
    shutil.copytree(root / "plans", cold / "plans")
    for name in ("config.json", "selected.json"):
        shutil.copy2(root / name, cold / name)
    options = json.loads((cold / "config.json").read_text())
    options["resume"] = str(root / "training/checkpoints/step_00000002.ssp")
    (cold / "config.json").write_text(json.dumps(options))
    worker = subprocess.run(
        [
            sys.executable,
            "-S",
            str(Path(__file__).with_name("_qwen35_lora_execute.py")),
            str(cold),
        ],
        env={
            **os.environ,
            "TRITON_CACHE_DIR": str(tmp_path / "cold-triton"),
            "TORCHINDUCTOR_CACHE_DIR": str(tmp_path / "cold-inductor"),
        },
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert worker.returncode == 0, worker.stdout + worker.stderr
    actual = manifest(cold, 3)
    assert {x["name"]: x["sha256"] for x in actual["tensors"]} == {
        x["name"]: x["sha256"] for x in expected["tensors"]
    }
    for name, metric in (("metrics", "train/loss"), ("evaluation", "validation_loss")):
        original = [
            json.loads(line)
            for line in (root / f"training/{name}.jsonl").read_text().splitlines()
        ]
        resumed = [
            json.loads(line)
            for line in (cold / f"training/{name}.jsonl").read_text().splitlines()
        ]
        assert math.isfinite(original[-1][metric])
        assert resumed[-1][metric] == original[-1][metric]
