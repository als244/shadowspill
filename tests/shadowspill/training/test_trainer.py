"""Loop behavior independent of GPU code; real saved-plan replay is separate."""

import json
import random
import weakref
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from shadowspill import MemoryBudgets, Trainer, apply_profiles, plan_program
from shadowspill.execution import TrainingResult
from shadowspill.training import _state
from shadowspill.training.observations import MetricSummary, MetricTable
from tests.shadowspill.execution.test_checkpoint import MemoryExecutor
from tests.shadowspill.offline import _example


@pytest.fixture
def executor():
    program = replace(
        _example.program(),
        bindings={
            "kind": "step",
            "optimizer_bindings": [{"name": "lr", "role": "hyperparameter"}],
        },
    )
    plan = plan_program(
        apply_profiles(program, _example.profiles()),
        memory_budgets=MemoryBudgets(100, 64),
        communication_info=_example.communication(),
        search_options=_example.options(),
    )
    result = MemoryExecutor()
    result.plan, result.device = plan, torch.device("cpu")
    result.calls, result.closed = [], False
    result.synchronize = lambda: None
    result.close = lambda: setattr(result, "closed", True)

    def run(**inputs):
        result.calls.append(inputs)
        output = TrainingResult(
            (torch.tensor(2.0),), ({"observed": torch.tensor(3.0)},)
        )
        return SimpleNamespace(result=lambda: output)

    result.run = run
    return result


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


class Schedule:
    def __init__(self):
        self.scale = torch.tensor(0.2)

    def __call__(self, step):
        return self.scale / (step + 1)

    def state_dict(self):
        return {"scale": self.scale}

    def load_state_dict(self, values):
        self.scale = values["scale"]


class Source:
    def __init__(self):
        self.position = 0

    def __iter__(self):
        while self.position < 3:
            self.position += 1
            yield {"args": [(self.position,)]}

    def state_dict(self):
        return {"position": self.position}

    def load_state_dict(self, value):
        self.position = value["position"]


def test_resume_restores_source_schedule_rng_and_checks_before_mutation(
    executor, tmp_path
):
    schedule, source = Schedule(), Source()
    trainer = Trainer(executor.plan, executor=executor, schedules={"lr": schedule})
    trainer.fit(source, steps=1, log_every=0)
    trainer.save(tmp_path / "resume.ssp", chunk_bytes=20)
    expected = random.random(), torch.rand(3), np.random.rand(3)
    executor.values["step"].zero_()
    schedule.scale.zero_()
    resumed = Trainer(executor.plan, executor=executor, schedules={"lr": Schedule()})
    resumed.load(tmp_path / "resume.ssp", chunk_bytes=28)
    assert resumed.step_count == 1 and executor.values["step"].item() == 3
    assert resumed.schedules["lr"].scale.item() == pytest.approx(0.2)
    assert random.random() == expected[0]
    torch.testing.assert_close(torch.rand(3), expected[1])
    np.testing.assert_equal(np.random.rand(3), expected[2])
    with pytest.raises(ValueError, match="resumable source"):
        resumed.fit([], steps=3)
    fresh_source = Source()
    resumed.fit(fresh_source, steps=3, log_every=0)
    assert fresh_source.position == 3
    assert [call["args"][0][0] for call in executor.calls] == [1, 2, 3]
    bad = Trainer(executor.plan, executor=executor)
    executor.values["step"].fill_(99)
    with pytest.raises(ValueError, match="schedules differ"):
        bad.load(tmp_path / "resume.ssp")
    assert executor.values["step"].item() == 99


def test_loop_state_codec_preserves_types_without_pickle():
    value = {
        1: (torch.ones(3, dtype=torch.bfloat16), np.arange(4, dtype=np.uint32)),
        "empty": torch.empty(0, 3),
        "nested": [False, None, {2: "text"}],
    }
    decoded = _state.decode(json.loads(json.dumps(_state.encode(value))))
    torch.testing.assert_close(decoded[1][0], value[1][0])
    np.testing.assert_equal(decoded[1][1], value[1][1])
    assert decoded["nested"] == value["nested"]
    assert decoded["empty"].shape == (0, 3)
    with pytest.raises(TypeError, match="object arrays"):
        _state.encode(np.array([{}], dtype=object))
