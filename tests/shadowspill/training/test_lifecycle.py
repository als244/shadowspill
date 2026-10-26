"""Trainer owns only the Executor it constructs; failures preserve ownership."""

import weakref
from types import SimpleNamespace

import pytest
import torch

from shadowspill import Trainer
from shadowspill.execution import TrainingResult


@pytest.mark.parametrize("borrowed", [False, True])
def test_initialization_failure_releases_only_owned_executor(
    executor, monkeypatch, borrowed
):
    import shadowspill.training.trainer as module

    closed, synchronized = [], []
    executor.close = lambda: closed.append(True)
    executor.synchronize = lambda: synchronized.append(True)

    def fail(*args, **kwargs):
        raise ValueError("initializer failed")

    executor.initialize = fail
    monkeypatch.setattr(module, "Executor", lambda *a, **kw: executor)
    trainer = Trainer(executor.plan, executor=executor if borrowed else None)
    with pytest.raises(ValueError, match="initializer failed"), trainer:
        trainer.initialize(lambda request, chunk: chunk.zero_())
    trainer.close()
    assert closed == ([] if borrowed else [True])
    assert synchronized == ([True] if borrowed else [])
    with pytest.raises(RuntimeError, match="closed"):
        trainer.initialize(lambda *_: None)


def test_failed_dispatch_does_not_advance_update_count(executor):
    def fail(**kwargs):
        raise RuntimeError("execution failed")

    executor.run = fail
    with Trainer(executor.plan, executor=executor) as trainer:
        with pytest.raises(RuntimeError, match="execution failed"):
            trainer.step(args=[()])
        assert trainer.step_count == 0


def test_cleanup_failure_does_not_hide_training_error(executor, monkeypatch):
    import shadowspill.training.trainer as module

    def fail():
        raise RuntimeError("cleanup failed")

    executor.close = fail
    monkeypatch.setattr(module, "Executor", lambda *a, **kw: executor)
    with (
        pytest.raises(ValueError, match="training failed") as caught,
        Trainer(executor.plan),
    ):
        raise ValueError("training failed")
    assert any("cleanup failed" in note for note in caught.value.__notes__)


def test_owned_executor_and_predispatch_validation(executor, monkeypatch):
    import shadowspill.training.trainer as module

    opened = []

    def create(plan, **kwargs):
        opened.append(kwargs)
        return executor

    monkeypatch.setattr(module, "Executor", create)
    with pytest.raises(ValueError, match="unknown runtime"):
        Trainer(executor.plan, schedules={"other": lambda step: 0.1})
    with pytest.raises(TypeError, match="Plan produced"):
        Trainer(torch.nn.Linear(2, 2))
    assert not opened
    with Trainer(executor.plan) as trainer:
        with pytest.raises(ValueError, match="cadences"):
            trainer.fit([], steps=1, log_every=-1)
        with pytest.raises(TypeError, match="source item"):
            trainer.fit([torch.ones(2)], steps=1)
    assert executor.closed and len(opened) == 1


def test_device_results_are_released_before_calling_a_reducer(executor):
    outputs = []

    def run(**inputs):
        loss = torch.tensor(2.0)
        outputs.append(weakref.ref(loss))
        return SimpleNamespace(result=lambda: TrainingResult((loss,), ()))

    def fail(observations):
        assert observations.losses == (2.0,)
        assert all(output() is None for output in outputs)
        raise ValueError("reducer failed")

    executor.run = run
    with Trainer(executor.plan, executor=executor, metric_reducer=fail) as trainer:
        with pytest.raises(ValueError, match="reducer failed"):
            trainer.step(args=[()])
        assert trainer.step_count == 1
