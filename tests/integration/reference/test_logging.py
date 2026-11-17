"""Logger uses the actual completed update index for scalars and tables."""

from __future__ import annotations

import json

import pytest

from reference.training.logger import Wandb
from shadowspill.training.observations import MetricTable


@pytest.mark.parametrize("first_step", [1, 500])
def test_wandb_uses_completed_updates_for_scalars_tables_and_eval(
    tmp_path, monkeypatch, first_step
):
    pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_SILENT", "true")
    with Wandb(
        project="shadowspill-logger-test", run_dir=tmp_path, mode="offline"
    ) as logger:
        for step in range(first_step, first_step + 3):
            logger({"step": step, "train/loss": 1 / step, "train/step_seconds": 0.3})
            logger({"step": step, "packing/trained_tokens": 1024})
            logger.table(
                step,
                "parameters/norms",
                MetricTable(("parameter", "norm"), (("weight", 2.0),)),
            )
            if step == first_step + 1:
                logger({"step": step, "eval/loss": 0.5})
            assert logger.run.step == step
    record = json.loads((tmp_path / "wandb_run.json").read_text())
    assert record["project"] == "shadowspill-logger-test"
    assert record["id"] == logger.run.id and "url" in record
