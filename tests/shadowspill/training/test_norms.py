"""Optional eager diagnostics respect cadence and observer contracts."""

import pytest

from shadowspill import Trainer


def test_parameter_norm_cadence_feeds_existing_reducer_and_callbacks(
    executor, tmp_path
):
    seen, logged = [], []
    with Trainer(
        executor.plan,
        executor=executor,
        parameter_norms_every=2,
        metric_reducer=lambda values: (
            {"weight_norm": values.parameter_metrics["weight"]["param_norm"].item()}
            if values.parameter_metrics
            else {}
        ),
    ) as trainer:
        trainer.fit(
            [{"args": [()]} for _ in range(3)],
            steps=3,
            run_dir=tmp_path,
            logger=logged.append,
            callbacks=[lambda owner, row: seen.append(row.parameter_metrics)],
        )
    assert [call["collect_parameter_norms"] for call in executor.calls] == [
        False,
        True,
        False,
    ]
    assert not seen[0] and not seen[2]
    assert seen[1]["weight"]["grad_norm"].item() == 5
    assert logged[1]["train/weight_norm"] == 4
    assert "train/weight_norm" not in logged[0]


def test_parameter_norms_require_enabled_executor(executor):
    executor.parameter_norms = False
    with pytest.raises(ValueError, match="enable parameter_norms"):
        Trainer(executor.plan, executor=executor, parameter_norms_every=1)
    for value in (-1, True, 1.5):
        with pytest.raises(ValueError, match="nonnegative integer"):
            Trainer(executor.plan, executor=executor, parameter_norms_every=value)
