"""W&B's public step counter must follow optimizer steps, including tables."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.metrics import Logger
from training.observations import MetricTable


@pytest.mark.parametrize("first_step", [0, 500])
def test_wandb_uses_training_steps_for_scalars_tables_and_eval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, first_step: int
) -> None:
    pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_SILENT", "true")
    logger = Logger(tmp_path, {}, project="shadowspill-logger-test", mode="offline")
    try:
        logger.log(first_step, echo=False, setup_seconds=1.0)
        for step in range(first_step, first_step + 3):
            logger.log(step, echo=False, loss=1.0 / (step + 1))
            logger.log(step, echo=False, **{"packing/trained_tokens": 1024})
            logger.table(
                step,
                "parameters/norms",
                MetricTable(("parameter", "norm"), (("weight", 2.0),)),
            )
            if step == first_step + 1:
                logger.log(step, echo=False, val_loss=0.5)
            assert logger.wandb.step == step
    finally:
        logger.close()

    # Local logging still flushes every call and uses the same step values.
    metrics = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text().splitlines()
    ]
    assert [row["step"] for row in metrics if "loss" in row] == list(
        range(first_step, first_step + 3)
    )
    assert [row["step"] for row in metrics if "val_loss" in row] == [first_step + 1]


def test_elapsed_seconds_continue_from_first_record_when_logger_reopens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    readings = iter((100.0, 125.0, 180.0))
    monkeypatch.setattr(
        "training.metrics.time", SimpleNamespace(time=lambda: next(readings))
    )
    logger = Logger(tmp_path, {}, project=None, mode="offline")
    logger.log(0, echo=False, loss=2.0)
    logger.log(1, echo=False, loss=1.0)
    logger.close()
    resumed = Logger(tmp_path, {}, project=None, mode="offline")
    resumed.log(2, echo=False, loss=0.5)
    resumed.close()
    rows = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text().splitlines()
    ]
    assert [row["elapsed_seconds"] for row in rows] == [0.0, 25.0, 80.0]
