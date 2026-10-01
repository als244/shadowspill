"""Behavioral checks against an ordinary PyTorch loop on non-text data."""

from __future__ import annotations

import copy
import json

import pytest
import torch
from torch import nn

from shadowspill.training import Forward, Trainer, reset_parameters
from shadowspill.training.backends import PyTorch
from shadowspill.training.observations import parameter_norms
from shadowspill.training.schedules import WarmupCosine


def objective(model, batch):
    error = model(batch["features"]) - batch["target"]
    return error.square().mean(), {"error_sum": error.detach().sum()}


def data(seed=0):
    gen = torch.Generator().manual_seed(seed)
    return {
        "features": torch.randn(5, 3, generator=gen),
        "target": torch.randn(5, 2, generator=gen),
    }


def split(batch):
    for start, stop in ((0, 2), (2, 5)):
        yield (
            {name: value[start:stop] for name, value in batch.items()},
            (stop - start) / 5,
        )


def test_initialized_weights_and_unequal_microbatches_match_unsplit_update():
    torch.manual_seed(10)
    model = nn.Linear(3, 2)
    reference = copy.deepcopy(model)
    initial = {n: p.detach().clone() for n, p in model.named_parameters()}
    batch = data()
    with Trainer(
        model,
        objective=objective,
        optimizer=torch.optim.SGD,
        optimizer_args={"lr": 0.1},
        microbatches=split,
        parameter_metrics=parameter_norms,
        backend=PyTorch(compile=False, device="cpu"),
    ) as trainer:
        trainer.prepare(batch)
        for name, value in initial.items():
            torch.testing.assert_close(trainer.model.get_parameter(name), value)
        result = trainer.step(batch)
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    expected, _ = objective(reference, batch)
    expected.backward()
    optimizer.step()
    for found, want in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(found, want)
    assert result.step == 1
    assert result.loss == pytest.approx(expected.item(), rel=1e-6)
    assert len(result.metrics) == 2
    assert result.parameter_metrics["weight"]["grad_norm"].device.type == "cpu"


def test_tuple_schedule_broadcasts_to_groups_and_keeps_group_constants():
    model = nn.Linear(3, 2)

    def groups(m):
        return [
            {"params": [m.weight], "weight_decay": 0.25},
            {"params": [m.bias], "weight_decay": 0.0},
        ]

    with Trainer(
        model,
        objective=objective,
        optimizer=torch.optim.AdamW,
        optimizer_args={"lr": 0.01},
        parameter_groups=groups,
        schedules={"betas": lambda step: (0.8, 0.91)},
        backend=PyTorch(compile=False, device="cpu"),
    ) as trainer:
        trainer.prepare(data())
        trainer.step(data())
        actual = trainer._execution.optimizer.param_groups
        assert [g["betas"] for g in actual] == [(0.8, 0.91)] * 2
        assert [g["weight_decay"] for g in actual] == [0.25, 0.0]
        with pytest.raises(ValueError, match="undeclared"):
            trainer.step(data(), hyperparams={"lr": 0.2})


def test_forward_and_evaluation_preserve_training_mode_and_use_updated_state():
    torch.manual_seed(11)
    model = nn.Sequential(nn.Linear(3, 3), nn.Dropout(0.5), nn.Linear(3, 2))
    backend = PyTorch(compile=False, device="cpu")
    trainer = Trainer(
        model,
        objective=objective,
        optimizer=torch.optim.SGD,
        optimizer_args={"lr": 0.1},
        backend=backend,
    ).prepare(data())
    trainer.step(data())
    with Forward(
        trainer.model,
        forward_fn=lambda m, x: {"prediction": m(x["features"])},
        backend=backend,
    ) as forward:
        forward.prepare(data())
        actual = forward(data())["prediction"]
        assert model.training
        model.eval()
        expected = model(data()["features"])
        model.train()
        torch.testing.assert_close(actual, expected)
    evaluated = trainer.evaluate([data()], batches=1)
    assert evaluated.mean_loss >= 0
    assert model.training and model[1].training
    trainer.close()


def test_meta_requires_explicit_initialization_and_keeps_ties():
    class Tied(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(3, 2, bias=False)
            self.b = nn.Linear(3, 2, bias=False)
            self.b.weight = self.a.weight

        def forward(self, value):
            return self.a(value) + self.b(value)

    with torch.device("meta"):
        model = Tied()
    trainer = Trainer(
        model,
        objective=objective,
        optimizer=torch.optim.SGD,
        optimizer_args={"lr": 0.1},
        backend=PyTorch(compile=False, device="cpu"),
    )
    with pytest.raises(ValueError, match="meta model requires"):
        trainer.prepare(data())
    trainer.prepare(data(), initialize=reset_parameters)
    assert trainer.model.a.weight is trainer.model.b.weight
    trainer.step(data())
    trainer.close()


class Source:
    def __init__(self):
        self.position = 0

    def __iter__(self):
        return self

    def __next__(self):
        result = data(self.position)
        self.position += 1
        return result

    def state_dict(self):
        return {"position": self.position}

    def load_state_dict(self, state):
        self.position = state["position"]


def test_checkpoint_resume_matches_uninterrupted_run_and_logs_actual_steps(tmp_path):
    torch.manual_seed(5)
    start = nn.Linear(3, 2)

    def build(model):
        return Trainer(
            model,
            objective=objective,
            optimizer=torch.optim.AdamW,
            schedules={"lr": WarmupCosine(0.01, 0.001, 1, 4)},
            backend=PyTorch(compile=False, device="cpu"),
        )

    with build(copy.deepcopy(start)) as uninterrupted:
        uninterrupted.prepare(data())
        uninterrupted.fit(Source(), steps=4, log_every=0)
        expected = copy.deepcopy(uninterrupted.model.state_dict())
    with build(copy.deepcopy(start)) as first:
        first.prepare(data())
        first.fit(Source(), steps=2, run_dir=tmp_path / "run", checkpoint_every=2)
    checkpoint = tmp_path / "run/checkpoints/step_00000002"
    with build(copy.deepcopy(start)) as resumed:
        resumed.prepare(data(), checkpoint=checkpoint)
        source = Source()
        resumed.fit(source, steps=4, run_dir=tmp_path / "run")
        assert source.position == 4 and resumed.step_count == 4
        for name, found in resumed.model.state_dict().items():
            torch.testing.assert_close(found, expected[name], rtol=0, atol=0)
    records = [
        json.loads(line)
        for line in (tmp_path / "run/metrics.jsonl").read_text().splitlines()
    ]
    assert [r["step"] for r in records] == [1, 2, 3, 4]


def test_compiled_nested_inputs_accept_new_loss_scale_values():
    model = nn.Linear(3, 2)
    reference = copy.deepcopy(model)

    def scaled(update):
        yield update["batch"], update["scale"]

    trainer = Trainer(
        model,
        objective=objective,
        optimizer=torch.optim.SGD,
        optimizer_args={"lr": 0.02},
        microbatches=scaled,
        backend=PyTorch(compile=True, device="cpu"),
    )
    trainer.prepare({"batch": data(), "scale": 0.5})
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.02)
    for scale in (0.5, 0.125):
        trainer.step({"batch": data(), "scale": scale})
        loss, _ = objective(reference, data())
        (loss * scale).backward()
        optimizer.step()
        optimizer.zero_grad()
    for found, want in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(found, want)
    trainer.close()


def test_core_tuple_hyperparameters_broadcast_to_each_group():
    from shadowspill.pytorch.callables import _apply_hyperparams

    groups = [{"betas": (torch.tensor(0.9), torch.tensor(0.99))} for _ in range(2)]
    _apply_hyperparams(nn.Linear(1, 1), groups, {"betas": (0.8, 0.95)}, lambda _: None)
    for group in groups:
        assert [value.item() for value in group["betas"]] == pytest.approx([0.8, 0.95])
    with pytest.raises(ValueError, match="per group"):
        _apply_hyperparams(nn.Linear(1, 1), groups, {"betas": (0.7,)}, lambda _: None)
    for group in groups:
        assert [value.item() for value in group["betas"]] == pytest.approx([0.8, 0.95])


def test_core_forward_callback_keeps_state_names_and_original_model_call():
    from shadowspill.pytorch.planning.forward.capture import _select_forward

    model = nn.Sequential(nn.Linear(3, 2), nn.Sigmoid())
    original = copy.deepcopy(model)
    inputs = {"left": data()["features"], "right": data(2)["features"]}

    def callback(m, x):
        return {"sum": m(x["left"]) + m(x["right"])}

    _select_forward(model, callback)
    exported = torch.export.export(model, (inputs,), strict=True)
    torch.testing.assert_close(exported.module()(inputs), callback(original, inputs))
    assert [
        item.target
        for item in exported.graph_signature.input_specs
        if item.kind.name == "PARAMETER"
    ] == ["0.weight", "0.bias"]


def test_scaled_tuple_objective_is_accepted_by_the_planning_capture():
    from shadowspill.pytorch.capture.aot import capture_training_objective
    from shadowspill.training._inputs import ScaledObjective

    model = nn.Linear(3, 2)
    capture = capture_training_objective(
        model, ScaledObjective(objective), (data(), torch.tensor(0.25))
    )
    assert capture.objective_schema.tensor_metric_positions == (0,)


def test_startup_diagnostics_reuse_first_data_without_logging_or_schedule_updates(
    tmp_path,
):
    """A diagnostic backend must not consume another source item or loop step."""
    source = Source()
    calls, logs, schedule_calls = [], [], []
    model = nn.Linear(3, 2)

    def schedule(step):
        schedule_calls.append(step)
        return 0.01

    with Trainer(
        model,
        objective=objective,
        optimizer=torch.optim.SGD,
        schedules={"lr": schedule},
        backend=PyTorch(compile=False, device="cpu"),
    ) as trainer:
        trainer.prepare(data())

        def diagnose(batches, directory, *, warmup):
            calls.append((batches, directory, warmup, trainer.step_count))
            torch.rand(5)  # Diagnostic-only random draws must not reach training.
            return {"traced_seconds": 0.1}

        trainer._execution.diagnose = diagnose
        before_rng = torch.get_rng_state()
        trainer.fit(
            source,
            steps=2,
            run_dir=tmp_path,
            logger=logs.append,
            startup_diagnostics=True,
        )
        assert source.position == trainer.step_count == 2
        assert schedule_calls == [0, 1]
        assert [item["step"] for item in logs] == [1, 2]
        assert torch.equal(torch.get_rng_state(), before_rng)
    assert len(calls) == 1
    batches, directory, warmup, step = calls[0]
    assert directory == tmp_path / "startup" and warmup == 1 and step == 0
    torch.testing.assert_close(batches[0][0]["features"], data()["features"])
    assert len((tmp_path / "metrics.jsonl").read_text().splitlines()) == 2
