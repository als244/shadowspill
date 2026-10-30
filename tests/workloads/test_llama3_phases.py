"""The saved-plan text recipe keeps initialization bounded and data resumable."""

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

from shadowspill.execution.tensors import TensorChunk
from workloads.models.llama3.config import LLAMA_3_1_ROPE_SCALING, Config
from workloads.models.shared.decoder import RotaryEmbedding
from workloads.recipes.text.data.calls import Inputs
from workloads.recipes.text.initialization import Initialize


@pytest.mark.parametrize("scaling", [None, LLAMA_3_1_ROPE_SCALING])
def test_initializer_chunks_preserve_tables_norms_and_random_values(scaling):
    config = Config(1, 128, 2, 1, 256, 512, max_seq_len=64, rope_scaling=scaling)
    initialize = Initialize("llama3", config, seed=123)
    initialize.block_elements = 257  # Cross RNG blocks with deliberately odd chunks.

    def fill(name, shape, dtype, width):
        target = torch.empty(math.prod(shape), dtype=dtype)
        for start in range(0, target.numel(), width):
            request = TensorChunk(name, shape, dtype, start, target.numel())
            initialize(request, target[start : start + width])
        return target.reshape(shape)

    rotary = RotaryEmbedding(
        config.head_dim,
        base=config.rope_base,
        capacity=config.max_seq_len,
        scaling=scaling,
    )
    for name, shape, dtype in (
        ("embed.weight", (512, 128), torch.bfloat16),
        ("blocks.0.attn.wq.weight", (128, 128), torch.bfloat16),
        ("final_norm.weight", (128,), torch.bfloat16),
        ("rotary.cosine", (64, 64), torch.float32),
        ("rotary.sine", (64, 64), torch.float32),
    ):
        whole = fill(name, shape, dtype, math.prod(shape))
        chunked = fill(name, shape, dtype, 113)
        torch.testing.assert_close(whole, chunked, rtol=0, atol=0)
        if name.startswith("rotary"):
            torch.testing.assert_close(whole, getattr(rotary, name.split(".")[1]))
        elif "norm" in name:
            assert torch.equal(whole, torch.ones_like(whole))
        else:
            # Hugging Face's N(0, initializer_range) for embeddings and
            # projections; another seed draws other values.
            assert 0.018 < whole.float().std().item() < 0.022
            other = Initialize("llama3", config, seed=124)
            target = torch.empty(math.prod(shape), dtype=dtype)
            other(TensorChunk(name, shape, dtype, 0, target.numel()), target)
            assert not torch.equal(whole, target.reshape(shape))


def test_inputs_use_normalized_calls_and_restore_next_example(tmp_path):
    values = np.arange(513, dtype=np.uint32) % 32
    values.tofile(tmp_path / "train.bin")
    (values + 1).tofile(tmp_path / "val.bin")
    (tmp_path / "meta.json").write_text(json.dumps({"vocab_size": 64}))
    options = {
        "data": str(tmp_path),
        "sequence_length": 16,
        "sequences_per_step": 4,
        "sequences_per_example": 2,
    }
    source = Inputs(options)
    assert source.data.next_step == 0
    expected = source.example()
    assert len(expected["args"]) == 2
    assert all(
        k == {"seq_lens": (16, 16), "normalizer": 1 / 64} for k in expected["kwargs"]
    )
    torch.testing.assert_close(next(source), expected)
    state = source.state_dict()
    expected = next(source)
    restored = Inputs(options)
    restored.load_state_dict(state)
    torch.testing.assert_close(next(restored), expected)
    assert not torch.equal(
        source.example()["args"][0][0],
        Inputs(options, split="val").example()["args"][0][0],
    )


@pytest.mark.cuda
@pytest.mark.slow
def test_recipe_runs_with_saved_plans_after_original_caches_are_deleted(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    pytest.importorskip("mlops")
    data, root = tmp_path / "data", tmp_path / "run"
    data.mkdir()
    values = np.arange(4096, dtype=np.uint32) % 32
    values.tofile(data / "train.bin")
    ((values + 7) % 32).tofile(data / "val.bin")
    (data / "meta.json").write_text(json.dumps({"vocab_size": 512}))
    build_cache = tmp_path / "preparation-cache"
    environment = {
        **os.environ,
        "TRITON_CACHE_DIR": str(build_cache / "triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(build_cache / "inductor"),
    }
    command = [
        sys.executable,
        *(["-S"] if sys.flags.no_site else []),
        "-m",
        "workloads.recipes.text.train_llama3",
        "--data",
        str(data),
        "--run-dir",
        str(root),
        "--model",
        "llama3-tiny",
        "--sequence-length",
        "16",
        "--sequences-per-step",
        "4",
        "--sequences-per-example",
        "2",
        "--steps",
        "3",
        "--learning-rate",
        "0.001",
        "--checkpoint-every",
        "1",
        "--execution-gib",
        "2",
        "--spill-gib",
        "0.125",
        "--warmup-seconds",
        "0.001",
        "--measure-seconds",
        "0.002",
        "--no-plots",
    ]
    child = subprocess.run(
        command, env=environment, capture_output=True, text=True, timeout=600
    )
    logs = "\n".join(path.read_text() for path in (root / "logs").glob("*.log"))
    assert child.returncode == 0, child.stdout + child.stderr + logs
    metrics = [
        json.loads(line)
        for line in (root / "training/metrics.jsonl").read_text().splitlines()
    ]
    assert len(metrics) == 3
    assert metrics[-1]["train/loss"] < metrics[0]["train/loss"]
    assert (root / "reports/training/index.html").is_file()
    assert not list(root.rglob("*.cubin"))
    shutil.rmtree(build_cache)  # No original compiler or Triton cache remains.
    cold = tmp_path / "cold"
    cold.mkdir()
    (cold / "training").mkdir()
    shutil.copy2(root / "config.json", cold / "config.json")
    shutil.copytree(root / "plans", cold / "plans")
    hashes = []
    for mode in ("complete", "resume"):
        if mode == "resume":
            options = json.loads((cold / "config.json").read_text())
            options["resume"] = str(cold / "training/checkpoints/step_00000001.ssp")
            (cold / "config.json").write_text(json.dumps(options))
        worker = subprocess.run(
            [
                sys.executable,
                *(["-S"] if sys.flags.no_site else []),
                str(Path(__file__).with_name("_llama3_execute.py")),
                str(cold),
            ],
            env={
                **os.environ,
                "TRITON_CACHE_DIR": str(tmp_path / f"{mode}-triton"),
                "TORCHINDUCTOR_CACHE_DIR": str(tmp_path / f"{mode}-inductor"),
            },
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert worker.returncode == 0, worker.stdout + worker.stderr
        assert "Qualified cold text execution" in worker.stdout
        with zipfile.ZipFile(
            cold / "training/checkpoints/step_00000003.ssp"
        ) as archive:
            manifest = json.loads(archive.read("manifest.json"))
        hashes.append({e["name"]: e["sha256"] for e in manifest["tensors"]})
    assert hashes[0] == hashes[1]
    repeated = [
        json.loads(line)
        for line in (cold / "training/metrics.jsonl").read_text().splitlines()
    ]
    assert [m["train/loss"] for m in repeated] == [
        m["train/loss"] for m in (*metrics, *metrics[1:])
    ]
