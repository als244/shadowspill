"""Shipped configs resolve phase inputs and bounded initialization for every tensor."""

import json
import math
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import torch

from shadowspill.artifacts.tensors import TensorChunk
from shadowspill.training import config
from shadowspill.training._hyperparams import set_hyperparams
from shadowspill.training._model import build_optimizer
from workloads.recipes.text.experiment.hyperparameters import scheduled_arguments

CONFIGS = sorted(
    (Path(__file__).resolve().parents[2] / "training/configs").glob("*.json")
)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda path: path.stem)
def test_supplied_configs_resolve_bounded_values_and_runtime_schedule(path):
    options = json.loads(path.read_text())
    assert "backend" not in options and options["engine"] == "shadowspill"
    model = config.resolve(options["model"])
    initialize = config.resolve(options["initialize"])
    initialize.block_elements = 31
    assert initialize.config == model.config
    assert initialize.seed == options["seed"]
    for name, tensor in (*model.named_parameters(), *model.named_buffers()):
        assert tensor.is_meta
        count = tensor.numel()
        value = torch.empty(min(count, 17), dtype=tensor.dtype)
        request = TensorChunk(
            name, tuple(tensor.shape), tensor.dtype, count - value.numel(), count
        )
        initialize(request, value)
        assert torch.isfinite(value).all(), name
    optimizer = build_optimizer(
        config.resolve(options["optimizer"]),
        model.parameters(),
        scheduled_arguments(
            config.resolve(options["optimizer_args"]), options["schedules"]
        ),
    )
    rates = config.resolve(options["schedules"])
    values = {name: schedule(2) for name, schedule in rates.items()}
    held = optimizer.param_groups[0]["lr"]
    assert isinstance(held, torch.Tensor)
    set_hyperparams(model, optimizer, values)
    assert optimizer.param_groups[0]["lr"] is held
    assert held.item() == pytest.approx(values["optimizer_group.0.lr"])


def test_tensor_chunk_descriptor_does_not_import_execution_or_torch():
    subprocess.run(
        [
            sys.executable,
            *(["-S"] if sys.flags.no_site else []),
            "-c",
            """
import sys
from shadowspill import TensorChunk
assert 'torch' not in sys.modules
assert 'shadowspill.execution' not in sys.modules
assert TensorChunk('weight', (4,), 'float32', 0, 4).storage_elements == 4
""",
        ],
        check=True,
    )


@pytest.mark.cuda
@pytest.mark.parametrize(
    "family,lora",
    [("llama3", False), ("llama3", True), ("qwen35", False), ("olmoe", False)],
)
@pytest.mark.slow
def test_supplied_config_tiny_training_through_saved_phases(tmp_path, family, lora):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from tests.training._synthetic import write_dataset

    path = Path(__file__).resolve().parents[2] / f"training/configs/{family}_1b.json"
    options = json.loads(path.read_text())
    root = tmp_path / "run"
    options.update(
        run_dir=str(root),
        steps=3,
        max_seq_len=64,
        max_tokens_per_step=128,
        max_tokens_per_microbatch=64,
        execution_gib=4,
        spill_gib=0.5,
        eval_every=2,
        eval_microbatches=1,
        checkpoint_every=1,
        keep_last=2,
        wandb_project=None,
        warmup_seconds=0.001,
        measure_seconds=0.002,
        external_headroom_mib=768,
        plots=False,
    )
    options["model"].update(model=f"{family}-tiny", max_seq_len=64)
    options["initialize"].update(model=f"{family}-tiny", max_seq_len=64)
    options["schedules"]["optimizer_group.0.lr"].update(warmup_steps=1, total_steps=3)
    options["data"].update(
        directory=str(write_dataset(tmp_path / "data")), min_tokens_per_seq=16
    )
    if lora:
        options["model"]["lora"] = {"rank": 32, "alpha": 32}
    fixture = tmp_path / "config.json"
    fixture.write_text(json.dumps(options))
    worker = subprocess.run(
        [
            sys.executable,
            *(["-S"] if sys.flags.no_site else []),
            "-m",
            "training.train",
            str(fixture),
        ],
        env=os.environ,
        capture_output=True,
        text=True,
        timeout=900,
    )
    logs = "\n".join(p.read_text() for p in (root / "logs").glob("*.log"))
    assert worker.returncode == 0, worker.stdout + worker.stderr + logs
    rows = [
        json.loads(row) for row in (root / "metrics.jsonl").read_text().splitlines()
    ]
    train = [row for row in rows if "train/loss" in row]
    assert [row["step"] for row in train] == [1, 2, 3]
    assert all(math.isfinite(row["train/loss"]) for row in train)
    evaluation = [row for row in rows if "eval/loss" in row]
    assert evaluation and all(math.isfinite(row["eval/loss"]) for row in evaluation)
    tensors = []
    for step in (2, 3):
        with zipfile.ZipFile(root / f"checkpoints/step_{step:08d}.ssp") as archive:
            metadata = json.loads(archive.read("manifest.json"))
            tensors.append(
                {item["name"]: item["sha256"] for item in metadata["tensors"]}
            )
    names = [
        name
        for name, parameter in config.resolve(options["model"]).named_parameters()
        if parameter.requires_grad
    ]
    assert not lora or all(".lora_" in name for name in names)
    assert names and any(tensors[0][name] != tensors[1][name] for name in names)
    assert (root / "report/index.html").is_file()
