"""A real family checkpoint through bounded loading and the saved-Plan recipe."""

import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from shadowspill.execution.tensors import TensorChunk
from shadowspill_workloads.lora import configure_lora
from shadowspill_workloads.recipes.text.qwen35_lora.initialize import Initialize
from tests.workloads.test_checkpoints import built, huggingface_state, metadata


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


def test_random_public_initializer_needs_only_configuration(checkpoint, tmp_path):
    from shadowspill_workloads.recipes.text.qwen35_lora.adapter import (
        describe,
        initializer,
    )

    root, base = checkpoint
    (root / "model.safetensors").unlink()
    data = tmp_path / "data"
    data.mkdir()
    (data / "meta.json").write_text(json.dumps({"vocab_size": 32}))
    options = dict(
        checkpoint=root, data=data, initialization="random", seed=17, sequence_length=32
    )
    info = describe(options)
    assert info["initialization"] == "random"
    assert info["frozen_state_id"].startswith("random-v1:")
    expected = base.embed.weight
    request = TensorChunk(
        "embed.weight", tuple(expected.shape), expected.dtype, 0, expected.numel()
    )
    with initializer(options, info) as fill:
        whole = torch.empty(257, dtype=expected.dtype, device="cpu")
        fill(request, whole)
        piece = torch.empty(100, dtype=expected.dtype, device="cpu")
        fill(
            TensorChunk(
                request.name, request.shape, request.dtype, 31, request.storage_elements
            ),
            piece,
        )
    assert torch.equal(piece, whole[31:131])
    assert torch.isfinite(whole).all() and whole.count_nonzero() > 0


def manifest(root, step):
    with tarfile.open(root / f"training/checkpoints/step_{step:08d}.ssp") as archive:
        return json.loads(archive.extractfile("manifest.json").read())


@pytest.mark.cuda
@pytest.mark.slow
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
        *(["-S"] if sys.flags.no_site else []),
        "-m",
        "examples.train_qwen35_lora",
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
        "--microbatch-sizes",
        "1",
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
    for name in ("config.json", "selected.json", "model.json"):
        shutil.copy2(root / name, cold / name)
    options = json.loads((cold / "config.json").read_text())
    options["resume"] = str(root / "training/checkpoints/step_00000002.ssp")
    (cold / "config.json").write_text(json.dumps(options))
    worker = subprocess.run(
        [
            sys.executable,
            *(["-S"] if sys.flags.no_site else []),
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
    assert {x["name"]: x["blake3"] for x in actual["tensors"]} == {
        x["name"]: x["blake3"] for x in expected["tensors"]
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
