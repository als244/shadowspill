"""Checkpoint LoRA keeps its data, precision and resume contracts across phases."""

import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from qualification.numerical.metrics import tensor_metrics
from qualification.numerical.tolerances import (
    LOSS_ABSOLUTE_TOLERANCE,
    LOSS_RELATIVE_TOLERANCE,
    meets_tensor_tolerance,
)
from shadowspill.training.preparation.cli import arguments
from shadowspill.training.preparation.configuration import configure as configure_shared
from shadowspill_workloads.inputs.text.tokenizer import tokenizer_identity
from shadowspill_workloads.recipes.text.glm53_flash_lora.arguments import parser
from shadowspill_workloads.recipes.text.glm53_flash_lora.inputs import Inputs
from shadowspill_workloads.recipes.text.glm53_flash_lora.model import (
    checkpoint,
    describe,
)
from tests.workloads.test_glm53_checkpoint import _write


def configure(argv):
    phase, values = arguments(parser(), argv)
    return configure_shared(
        "shadowspill_workloads.recipes.text.glm53_flash_lora.adapter", values, phase
    )


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
    _, options = configure(recipe[1])
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
    _, options = configure(recipe[1])
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
    root, options = configure(recipe[1])
    _, replanned = configure(
        [
            "--run-dir",
            str(root),
            "--phase",
            "plan",
            "--execution-gib",
            "5",
        ]
    )
    assert replanned == {**options, "execution_gib": 5}
    with pytest.raises(ValueError, match="changed budgets"):
        configure(
            ["--run-dir", str(root), "--phase", "execute", "--execution-gib", "7"]
        )
    with pytest.raises(ValueError, match="cadence"):
        configure(["--run-dir", str(root), "--phase", "execute", "--nsys-steps", "3"])
    assert json.loads((root / "config.json").read_text()) == replanned


def manifest(root, step):
    with tarfile.open(root / f"training/checkpoints/step_{step:08d}.ssp") as archive:
        return json.loads(archive.extractfile("manifest.json").read())


def compare_replay(reference, actual, *, learning_rate, steps):
    """Hold fast-kernel replay to the numerical gate's existing state bounds."""
    records = {}
    with tarfile.open(reference) as left, tarfile.open(actual) as right:
        expected, observed = (
            json.loads(archive.extractfile("manifest.json").read())
            for archive in (left, right)
        )
        assert observed["state"] == expected["state"]
        entries = {item["name"]: item for item in observed["tensors"]}
        assert set(entries) == {item["name"] for item in expected["tensors"]}
        for before in expected["tensors"]:
            name = before["name"]
            after = entries[name]
            geometry = {
                key: value
                for key, value in before.items()
                if key not in {"blake3", "member"}
            }
            assert all(after[key] == value for key, value in geometry.items()), name
            values = tuple(
                torch.frombuffer(
                    bytearray(archive.extractfile(entry["member"]).read()),
                    dtype=getattr(torch, entry["dtype"].removeprefix("torch.")),
                ).as_strided(entry["shape"], entry["stride"], entry["storage_offset"])
                for archive, entry in ((left, before), (right, after))
            )
            exact = before["blake3"] == after["blake3"]
            if (
                not values[0].is_floating_point()
                or name.startswith("optimizer_group.")
                or name.endswith(".step")
            ):
                records[name] = {"passed": exact, "bitwise_equal": exact}
            else:
                metric = tensor_metrics(*values)
                category = "optimizer" if name.startswith("optimizer.") else "model"
                passed = meets_tensor_tolerance(
                    metric,
                    key=f"state/{category}/{name}",
                    step_size=learning_rate,
                    steps=steps,
                )
                records[name] = {
                    "passed": passed,
                    "bitwise_equal": exact,
                    **asdict(metric),
                }
    return records


@pytest.mark.cuda
@pytest.mark.slow
def test_glm_checkpoint_training_evaluation_and_cold_resume(recipe, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    root, cli = recipe
    cache = tmp_path / "preparation-cache"
    entrypoint = [
        sys.executable,
        *(["-S"] if sys.flags.no_site else []),
        "-m",
        "examples.train_glm53_flash_lora",
    ]
    for phase in ("build", "profile", "plan", "execute"):
        arguments = cli if phase == "build" else ["--run-dir", str(root)]
        at = cache / phase
        child = subprocess.run(
            [*entrypoint, *arguments, "--phase", phase],
            env={
                **os.environ,
                "TRITON_CACHE_DIR": str(at / "triton"),
                "TORCHINDUCTOR_CACHE_DIR": str(at / "inductor"),
                "QUACK_CACHE_DIR": str(at / "quack"),
                "XDG_CACHE_HOME": str(at / "xdg"),
            },
            capture_output=True,
            text=True,
            timeout=900,
        )
        logs = "\n".join(path.read_text() for path in (root / "logs").glob("*.log"))
        assert child.returncode == 0, child.stdout + child.stderr + logs
    learning = json.loads((root / "learning.json").read_text())
    assert learning["relative_drop"] > 0
    completed = json.loads((root / "completed.json").read_text())
    assert completed["data_state"]["data"]["next_step"] == 4
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
            *(["-S"] if sys.flags.no_site else []),
            str(Path(__file__).with_name("_glm_lora_execute.py")),
            str(cold),
        ],
        env={
            **os.environ,
            "TRITON_CACHE_DIR": str(tmp_path / "cold-triton"),
            "TORCHINDUCTOR_CACHE_DIR": str(tmp_path / "cold-inductor"),
            "QUACK_CACHE_DIR": str(tmp_path / "cold-quack"),
            "XDG_CACHE_HOME": str(tmp_path / "cold-xdg"),
        },
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    comparisons = compare_replay(
        root / "training/checkpoints/step_00000004.ssp",
        cold / "training/checkpoints/step_00000004.ssp",
        learning_rate=options["learning_rate"],
        steps=4,
    )
    (tmp_path / "replay-comparison.json").write_text(json.dumps(comparisons, indent=2))
    assert not {
        name: value for name, value in comparisons.items() if not value["passed"]
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
        assert resumed[-1][metric] == pytest.approx(
            original[-1][metric],
            rel=LOSS_RELATIVE_TOLERANCE,
            abs=LOSS_ABSOLUTE_TOLERANCE,
        )
