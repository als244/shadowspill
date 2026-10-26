"""Checkpoint LoRA keeps its data, precision and resume contracts across phases."""

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

from tests.workloads.test_glm53_checkpoint import _write
from workloads.recipes.text.data.tokenizer import tokenizer_identity
from workloads.recipes.text.glm53_flash_lora.arguments import parser
from workloads.recipes.text.glm53_flash_lora.inputs import Inputs
from workloads.recipes.text.glm53_flash_lora.model import checkpoint, describe
from workloads.recipes.text.glm53_flash_lora.settings import configure


@pytest.fixture
def recipe(tmp_path):
    base, data, run = (tmp_path / name for name in ("base", "data", "run"))
    base.mkdir()
    data.mkdir()
    _write(base, fp8_experts=False)
    (base / "tokenizer.json").write_text('{"vocab": "fixture"}')
    (base / "tokenizer_config.json").write_text('{"eos_token": "end"}')
    tokens = np.arange(1025, dtype=np.uint32) % 128
    tokens.tofile(data / "train.bin")
    ((tokens + 7) % 128).tofile(data / "val.bin")
    (data / "meta.json").write_text(
        json.dumps(
            {
                "vocab_size": 128,
                "dtype": "uint32",
                "tokenizer_sha256": tokenizer_identity(base),
            }
        )
    )
    cli = [
        "--data",
        str(data),
        "--checkpoint",
        str(base),
        "--frozen-state-id",
        "tiny-glm-base",
        "--run-dir",
        str(run),
        "--sequence-length",
        "16",
        "--sequences-per-step",
        "2",
        "--steps",
        "4",
        "--schedule-steps",
        "4",
        "--repeat-training-batch",
        "--execution-gib",
        "6",
        "--spill-gib",
        "0.5",
        "--checkpoint-every",
        "1",
        "--search-workers",
        "1",
        "--external-headroom-mib",
        "768",
        "--warmup-seconds",
        "0.001",
        "--measure-seconds",
        "0.002",
        "--no-plots",
    ]
    return run, cli


def test_rank_local_calls_preserve_global_normalization_and_resume(recipe):
    _, options = configure(parser().parse_args(recipe[1]))
    options.update(sequences_per_example=1, ranks=2)
    seen = []
    for rank in range(2):
        source = Inputs(options, rank=rank, world_size=2)
        calls = next(source)
        assert len(calls["args"]) == 2
        assert (
            sum(
                args[0].numel() * kw["normalizer"]
                for args, kw in zip(calls["args"], calls["kwargs"], strict=True)
            )
            == 0.5
        )
        seen.append(calls["args"][0][0].tolist())
        state = source.state_dict()
        source.load_state_dict(state)
        assert source.data.next_step == 1
        for before, after in zip(calls["args"], next(source)["args"], strict=True):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert seen[0] != seen[1]


def test_checkpoint_metadata_rejects_changed_tokenizer(recipe):
    _, options = configure(parser().parse_args(recipe[1]))
    with checkpoint(options) as (model, source, _):
        metadata = describe(options, model, source)
        assert metadata["frozen_state_id"].startswith("tiny-glm-base:")
        assert model.config.expert_storage == "bf16"
        assert all(p.is_meta for p in model.parameters())
        assert all(
            "lora" in name for name, p in model.named_parameters() if p.requires_grad
        )
        (Path(options["checkpoint"]) / "tokenizer_config.json").write_text("{}")
        with pytest.raises(ValueError, match="tokenizer identity"):
            describe(options, model, source)


def test_cli_replan_preserves_preparation_settings_and_capture_validation(recipe):
    root, options = configure(parser().parse_args(recipe[1]))
    cli = parser()
    _, replanned = configure(
        cli.parse_args(
            [
                "--run-dir",
                str(root),
                "--phase",
                "plan",
                "--execution-gib",
                "5",
            ]
        )
    )
    assert replanned == {**options, "execution_gib": 5}
    with pytest.raises(ValueError, match="changed budgets"):
        configure(
            cli.parse_args(
                ["--run-dir", str(root), "--phase", "execute", "--execution-gib", "7"]
            )
        )
    with pytest.raises(ValueError, match="cadence"):
        configure(
            cli.parse_args(
                ["--run-dir", str(root), "--phase", "execute", "--nsys-steps", "3"]
            )
        )
    assert json.loads((root / "config.json").read_text()) == replanned


def test_recipe_planning_imports_no_tensor_framework():
    subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import sys; import workloads.recipes.text.glm53_flash_lora.plan; "
            "assert 'torch' not in sys.modules; "
            "assert 'workloads.models' not in sys.modules",
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        check=True,
    )


def manifest(root, step):
    with zipfile.ZipFile(root / f"training/checkpoints/step_{step:08d}.ssp") as archive:
        return json.loads(archive.read("manifest.json"))


@pytest.mark.cuda
def test_glm_checkpoint_training_evaluation_and_cold_resume(recipe, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    root, cli = recipe
    cache = tmp_path / "preparation-cache"
    environment = {
        **os.environ,
        "TRITON_CACHE_DIR": str(cache / "triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"),
    }
    child = subprocess.run(
        [
            sys.executable,
            "-S",
            "-m",
            "workloads.recipes.text.train_glm53_flash_lora",
            *cli,
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=900,
    )
    logs = "\n".join(path.read_text() for path in (root / "logs").glob("*.log"))
    assert child.returncode == 0, child.stdout + child.stderr + logs
    learning = json.loads((root / "learning.json").read_text())
    assert learning["relative_drop"] > 0
    assert json.loads((root / "completed.json").read_text())["next_data_step"] == 4
    assert (root / "reports/training/index.html").is_file()
    assert (root / "reports/evaluation/index.html").is_file()
    expected = manifest(root, 4)
    assert "embed_tokens.weight" in expected["omitted_parameters"]
    assert all("lora" not in name for name in expected["omitted_parameters"])
    assert not list(root.rglob("*.cubin"))
    shutil.rmtree(cache)
    cold = tmp_path / "cold"
    shutil.copytree(root / "plans", cold / "plans")
    for name in ("config.json", "selected.json", "model.json"):
        shutil.copy2(root / name, cold / name)
    options = json.loads((cold / "config.json").read_text())
    options["resume"] = str(root / "training/checkpoints/step_00000003.ssp")
    (cold / "config.json").write_text(json.dumps(options))
    child = subprocess.run(
        [
            sys.executable,
            "-S",
            str(Path(__file__).with_name("_glm_lora_execute.py")),
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
    assert child.returncode == 0, child.stdout + child.stderr
    actual = manifest(cold, 4)
    assert {x["name"]: x["sha256"] for x in actual["tensors"]} == {
        x["name"]: x["sha256"] for x in expected["tensors"]
    }
    for name, metric in (
        ("metrics", "train/global_mean_loss"),
        ("evaluation", "validation_loss"),
    ):
        original = [
            json.loads(row)
            for row in (root / f"training/{name}.jsonl").read_text().splitlines()
        ]
        resumed = [
            json.loads(row)
            for row in (cold / f"training/{name}.jsonl").read_text().splitlines()
        ]
        assert math.isfinite(original[-1][metric])
        assert resumed[-1][metric] == original[-1][metric]
