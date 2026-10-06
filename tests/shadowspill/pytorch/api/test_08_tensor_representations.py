"""Full planned updates, inference and checkpointing for physical tensor state."""

from __future__ import annotations

import copy
from fractions import Fraction

import pytest
import torch

from qualification.profiling import CORRECTNESS_PROFILING
from shadowspill.ir import TaskAlternativeChoice
from shadowspill.planner import SearchOptions
from shadowspill.planner.search.algorithms.pressurefit import (
    PressureFit,
    PressureFitOptions,
)
from shadowspill.training import Forward, Trainer
from shadowspill.training.backends import ShadowSpill
from tests.shadowspill.pytorch.state.representations import model

pytestmark = [pytest.mark.fresh_process, pytest.mark.cuda]


def test_planned_representation_save(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch, recompute=False)


def test_planned_representation_recompute(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch, recompute=True)


def test_planned_packed_parameter_gradients(tmp_path):
    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.Parameter(torch.randn(4, 8))
            self.second = torch.nn.Parameter(torch.randn(4, 8))

        def forward(self, value):
            return torch.nn.functional.linear(
                value, torch.cat((self.first, self.second))
            )

    reference = Network()
    source = copy.deepcopy(reference)
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.01, foreach=False)
    sample = torch.randn(3, 8)

    def objective(net, batch):
        return net(batch).square().mean()

    with (
        ShadowSpill(
            device="cuda:0",
            execution_gib=2,
            spill_gib=1,
            partition="whole",
            artifact_store=tmp_path / "artifacts",
            profiling_options=CORRECTNESS_PROFILING,
        ) as backend,
        Trainer(
            source,
            objective=objective,
            optimizer=torch.optim.SGD,
            optimizer_args={"lr": 0.01, "foreach": False},
            backend=backend,
        ) as trainer,
    ):
        trainer.prepare(sample)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            expected = objective(reference, sample)
            expected.backward()
            optimizer.step()
            result = trainer.step(sample)
            assert result.loss == pytest.approx(expected.item(), rel=2e-5, abs=1e-7)
        torch.testing.assert_close(
            trainer._execution.call.state_dict()["model"],
            reference.state_dict(),
            rtol=2e-5,
            atol=1e-6,
        )


def _run(tmp_path, monkeypatch, *, recompute):
    variant = "recompute" if recompute else "save"

    def choices(program, _shares):
        return (
            tuple(
                TaskAlternativeChoice(group.group_id, variant)
                for group in program.task_alternative_groups
            ),
        )

    # Small normal searches enumerate all resolutions. These regressions must
    # execute the variant named by the test, including the recompute entrypoint.
    monkeypatch.setattr(
        "shadowspill.planner.search.algorithms.pressurefit.resolutions", choices
    )
    reference = model()
    source = model()
    source.register_buffer("empty", torch.empty(0, 3))
    master = torch.nn.Parameter(reference.weight.dense().detach())
    optimizer = torch.optim.SGD([master], lr=0.01, foreach=False)
    sample = torch.arange(24, dtype=torch.float32).reshape(3, 8) / 24

    def objective(net, data):
        return net(data).square().mean()

    with (
        ShadowSpill(
            device="cuda:0",
            execution_gib=2,
            spill_gib=1,
            artifact_store=tmp_path / "artifacts",
            partition="whole",
            profiling_options=CORRECTNESS_PROFILING,
            search_options=SearchOptions(
                algorithm=PressureFit(
                    PressureFitOptions(resolution_options=(Fraction(recompute),))
                ),
            ),
        ) as backend,
        Trainer(
            source,
            objective=objective,
            optimizer=torch.optim.SGD,
            optimizer_args={"lr": 0.01, "foreach": False},
            master_dtype=torch.float32,
            grad_dtype=torch.float32,
            backend=backend,
        ) as trainer,
    ):
        trainer.prepare(sample)
        with Forward(trainer.model, backend=backend) as forward:
            forward.prepare(sample)
            for _ in range(3):
                expected = objective(reference, sample)
                master.grad = torch.autograd.grad(expected, reference.weight)[0]
                optimizer.step()
                with torch.no_grad():
                    reference.weight.copy_(master)
                result = trainer.step(sample)
                assert result.loss == pytest.approx(expected.item(), rel=2e-5, abs=1e-7)
                output = forward(sample)
                torch.testing.assert_close(
                    output.cpu(), reference(sample).detach(), rtol=2e-5, atol=1e-6
                )
                del output
        actual = trainer._execution.call.state_dict()["model"]["weight"]
        torch.testing.assert_close(actual, master.detach(), rtol=2e-5, atol=1e-6)
        assert trainer._execution.call.state_dict()["model"]["empty"].shape == (0, 3)
        checkpoint = trainer.save(tmp_path / "checkpoint")
        expected_loss = trainer.step(sample).loss
        trainer.load(checkpoint)
        assert trainer.step(sample).loss == pytest.approx(
            expected_loss, rel=1e-6, abs=1e-7
        )
        checkpoint = trainer.save(tmp_path / "compute-checkpoint", weights="compute")
        expected_loss = trainer.step(sample).loss
        trainer.load(checkpoint)
        assert trainer.step(sample).loss == pytest.approx(
            expected_loss, rel=1e-6, abs=1e-7
        )
