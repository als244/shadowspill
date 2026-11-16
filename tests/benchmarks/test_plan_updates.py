"""Preserve measured invocation ownership and caller-controlled update settings."""

from types import SimpleNamespace

import pytest
import torch

from benchmarks.plan_runner.api import _traced_inputs
from benchmarks.plan_runner.updates import updates
from shadowspill.execution import TrainingResult


def test_controls_and_owned_observations_survive_executor_release():
    invoked, scheduled = [], []
    value = torch.tensor(1.0)
    diagnostics = object()
    live = False

    def run(**inputs):
        nonlocal live
        live = True
        invoked.append(inputs)
        value.fill_(len(invoked))
        return SimpleNamespace(
            result=lambda: TrainingResult((value,), ({"metric": value},)),
            diagnostics=diagnostics,
        )

    def synchronize():
        nonlocal live
        value.fill_(-999)
        live = False

    executor = SimpleNamespace(
        plan=SimpleNamespace(
            program=SimpleNamespace(
                bindings={
                    "optimizer_bindings": [{"role": "hyperparameter", "name": "lr"}]
                }
            )
        ),
        run=run,
        synchronize=synchronize,
    )

    def schedule(step):
        scheduled.append(step)
        return 0.1 * (step + 1)

    def reduce(observations):
        assert not live
        assert observations.metrics[0]["metric"] == observations.losses[0]
        return {"loss": observations.losses[0]}

    override = {"lr": 0.5}

    def source():
        yield {"args": [(1,)], "hyperparams": override}
        yield {"args": [(2,)]}
        raise AssertionError("runner consumed input beyond the requested updates")

    result = list(
        updates(
            executor,
            _traced_inputs(source(), 2, True),
            steps=2,
            schedules={"lr": schedule},
            metric_reducer=reduce,
            parameter_norms_every=2,
            nsys=None,
            rank=0,
        )
    )
    assert scheduled == [0, 1]
    assert [row["hyperparams"] for row in invoked] == [{"lr": 0.5}, {"lr": 0.2}]
    assert override == {"lr": 0.5}
    assert [row["collect_parameter_norms"] for row in invoked] == [False, True]
    assert [row.loss for row in result] == [1, 2]
    assert [row.diagnostics for row in result] == [None, diagnostics]
    assert value.item() == -999 and not live


def test_exhausted_input_does_not_publish_a_complete_benchmark():
    executor = SimpleNamespace(
        plan=SimpleNamespace(program=SimpleNamespace(bindings={}))
    )
    with pytest.raises(RuntimeError, match="ended after 0 of 2"):
        next(
            updates(
                executor,
                (),
                steps=2,
                schedules=None,
                metric_reducer=None,
                parameter_norms_every=0,
                nsys=None,
                rank=0,
            )
        )
