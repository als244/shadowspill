"""Evaluation consumes an admitted forward Plan without advancing training."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from shadowspill import Trainer


def test_evaluation_requires_prebuilt_plan_and_does_not_advance_loop(executor):
    evaluation = replace(
        executor.plan,
        profiled_program=replace(
            executor.plan.profiled_program,
            program=replace(executor.plan.program, bindings={"kind": "forward"}),
        ),
    )
    with pytest.raises(TypeError, match="build_forward"):
        Trainer(executor.plan, executor=executor, evaluation_plan=executor.plan)
    with pytest.raises(ValueError, match="different evaluation Plan"):
        Trainer(executor.plan, executor=executor, evaluation_plan=evaluation)
    with (
        Trainer(executor.plan, executor=executor) as trainer,
        pytest.raises(RuntimeError, match="no prebuilt evaluation"),
    ):
        trainer.evaluate(args=[()])
    executor.plans["evaluation"] = evaluation
    scheduled = []
    trainer = Trainer(
        executor.plan,
        executor=executor,
        evaluation_plan=evaluation,
        schedules={"lr": lambda step: scheduled.append(step) or 0.1},
    )
    payload = {"predictions": torch.ones(2, 3), "label": "held out"}
    executor.run = lambda **kw: (
        executor.calls.append(kw) or SimpleNamespace(result=lambda: payload)
    )
    result = trainer.evaluate(args=[()])
    torch.testing.assert_close(result["predictions"], payload["predictions"])
    assert result["label"] == payload["label"]
    assert executor.calls == [
        {"plan": "evaluation", "args": [()], "kwargs": None, "input_loader": None}
    ]
    assert trainer.step_count == 0 and scheduled == []
