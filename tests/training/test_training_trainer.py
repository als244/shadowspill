"""The trainer end to end on the CPU: a tiny model on the PyTorch backend."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.training._synthetic import NOTES, write_dataset
from training.backends import Microbatch
from training.backends.pytorch import PyTorch
from training.trainer import Trainer


def _config(tmp_path: Path) -> Path:
    config = {
        "settings": [
            {"@call": "tests.training._synthetic:note", "text": "first"},
            {"@call": "tests.training._synthetic:note", "text": "second"},
        ],
        "model": {
            "@call": "training.models:build_on_meta",
            "model": "@tests.training._synthetic:TinyModel",
        },
        "objective": "@training.objectives:model_loss",
        "optimizer": "@torch.optim:AdamW",
        "optimizer_args": {"lr": 0.01},
        "data": {
            "@call": "training.data:PackedTokens",
            "directory": str(write_dataset(tmp_path / "tokens")),
        },
        "steps": 6,
        "max_seq_len": 512,
        "max_tokens_per_step": 4096,
        "max_tokens_per_microbatch": 2048,
        "schedule": {
            "@call": "training.schedules:WarmupCosine",
            "lr": 0.01,
            "min_lr": 0.001,
            "warmup_steps": 2,
        },
        "backend": {
            "@call": "training.backends.pytorch:PyTorch",
            "compile": False,
            "device": "cpu",
        },
        "eval_every": 3,
        "eval_batches": 2,
        "checkpoint_every": 3,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


def _records(run_dir: Path, key: str) -> dict[int, dict]:
    found = {}
    for line in (run_dir / "metrics.jsonl").read_text().splitlines():
        record = json.loads(line)
        if key in record:
            found[record["step"]] = record
    return found


def test_a_run_trains_logs_and_resumes_where_it_stopped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _config(tmp_path)
    whole = tmp_path / "whole"
    NOTES.clear()
    trainer = Trainer.from_config(config, [f"run_dir={whole}"])
    assert NOTES == ["first", "second"]  # the config's settings, in order
    last = trainer.train()
    steps = _records(whole, "loss")
    packing = _records(whole, "packing/sequences")
    assert sorted(steps) == sorted(packing) == list(range(6))
    for step in range(6):
        trained = packing[step]["packing/trained_tokens"]
        assert steps[step]["tokens_per_second"] == trained / steps[step]["step_seconds"]
    assert steps[5]["loss"] < steps[0]["loss"]
    assert last["val_loss"] == _records(whole, "val_loss")[5]["val_loss"]
    printed = [
        line for line in capsys.readouterr().out.splitlines() if "| loss" in line
    ]
    assert len(printed) == 6  # one line a step; what was packed stays off stdout
    assert json.loads((whole / "config.json").read_text())["steps"] == 6

    resumed = tmp_path / "resumed"
    Trainer.from_config(config, [f"run_dir={resumed}", "steps=3"]).train()
    Trainer.from_config(config, [f"run_dir={resumed}"]).train()
    again = _records(resumed, "loss")
    for step in range(6):
        assert again[step]["loss"] == steps[step]["loss"]
        assert again[step]["lr"] == steps[step]["lr"]
    again_packing = _records(resumed, "packing/sequences")
    for step in range(6):
        for name, value in packing[step].items():
            if name.startswith("packing/"):
                assert again_packing[step][name] == value


class _Recorded(PyTorch):
    """The CPU backend, noting what the trainer asks of it, in order."""

    def __init__(self) -> None:
        super().__init__(compile=False, device="cpu")
        self.calls: list[str] = []

    def step(self, microbatches: list[Microbatch], lr: float | None) -> list[float]:
        self.calls.append("step")
        return super().step(microbatches, lr)

    def synchronize(self) -> None:
        self.calls.append("synchronize")
        super().synchronize()

    def evaluate(self, microbatches: list[Microbatch]) -> list[float]:
        self.calls.append("evaluate")
        return super().evaluate(microbatches)

    def save(self, path: Path) -> None:
        self.calls.append("save")
        super().save(path)


def test_a_step_is_finished_before_it_is_evaluated_or_saved(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    overrides = [f"run_dir={run_dir}", "steps=4", "eval_every=2", "checkpoint_every=4"]
    trainer = Trainer.from_config(_config(tmp_path), overrides)
    trainer.backend = backend = _Recorded()
    trainer.train()

    assert backend.calls == [
        "step", "step", "synchronize", "evaluate",
        "step", "step", "synchronize", "evaluate", "save",
    ]  # fmt: skip
    # A step's line is written when the next step starts, or once it has
    # finished when an evaluation follows: always before that evaluation's.
    lines = [json.loads(line) for line in (run_dir / "metrics.jsonl").open()]
    order = [
        (line["step"], "val_loss" if "val_loss" in line else "loss")
        for line in lines
        if "loss" in line or "val_loss" in line
    ]
    assert order == [
        (0, "loss"), (1, "loss"), (1, "val_loss"),
        (2, "loss"), (3, "loss"), (3, "val_loss"),
    ]  # fmt: skip


def test_setting_up_on_pytorch_plans_nothing(tmp_path: Path) -> None:
    trainer = Trainer.from_config(_config(tmp_path), [f"run_dir={tmp_path / 'run'}"])
    assert trainer.setup() is None
    assert trainer.plan is None and trainer.planning is None
    assert (trainer.tokens, trainer.microbatches) == (2048, 2)
    trainer.close()


def test_token_counts_must_fit_the_data(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with pytest.raises(ValueError, match="multiple of max_seq_len"):
        Trainer.from_config(config, ["run_dir=x", "max_tokens_per_step=1000"])
    with pytest.raises(ValueError, match="must divide"):
        Trainer.from_config(config, ["run_dir=x", "max_tokens_per_microbatch=3072"])
