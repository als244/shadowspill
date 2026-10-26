"""Fit owns update counts, schedules, callbacks and log cadence."""

import json

import pytest

from shadowspill import Trainer
from shadowspill.training.observations import MetricSummary, MetricTable


def test_fit_schedules_logs_checkpoints_and_borrowed_executor(executor, tmp_path):
    from shadowspill.training import StepResult

    observed = []
    summary = MetricSummary(
        {"units": 7.0}, {"detail": MetricTable(("loss",), ((2.0,),))}
    )
    with Trainer(
        executor.plan,
        executor=executor,
        schedules={"lr": lambda step: 0.1 / (step + 1)},
        metric_reducer=lambda values: summary,
    ) as trainer:
        last = trainer.fit(
            [{"args": [()]} for _ in range(3)],
            steps=3,
            run_dir=tmp_path,
            callbacks=[lambda owner, row: observed.append(row.step)],
            checkpoint_every=1,
            keep_last=2,
            tables_every=2,
        )
        assert isinstance(last, StepResult) and last.loss == 2.0
        assert last.metrics[0]["observed"].device.type == "cpu"
        assert trainer.fit([], steps=3) is None
        with pytest.raises(RuntimeError, match="source ended"):
            trainer.fit([], steps=4)
    assert not executor.closed
    assert observed == [1, 2, 3]
    assert [c["hyperparams"]["lr"] for c in executor.calls] == [0.1, 0.05, 0.1 / 3]
    assert [p.name for p in sorted((tmp_path / "checkpoints").iterdir())] == [
        "step_00000002.ssp",
        "step_00000003.ssp",
    ]
    rows = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text().splitlines()
    ]
    assert [row["step"] for row in rows] == [1, 2, 3]
    assert rows[-1]["train/units"] == 7.0
    assert len((tmp_path / "observations.jsonl").read_text().splitlines()) == 2
    with pytest.raises(RuntimeError, match="closed"):
        trainer.step(args=[()])
