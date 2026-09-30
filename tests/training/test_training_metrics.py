"""W&B's public step counter must follow optimizer steps, including tables."""

from __future__ import annotations

import json
from pathlib import Path

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
