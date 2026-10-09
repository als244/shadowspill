"""Run in a fresh process after the unchanged qualification baseline."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from qualification.profiling import CORRECTNESS_PROFILING
from shadowspill.training import Forward, Trainer
from shadowspill.training.backends import ShadowSpill

pytestmark = [pytest.mark.fresh_process, pytest.mark.cuda]


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(6, 8)
        self.norm = nn.LayerNorm(8)
        self.last = nn.Linear(8, 3)

    def forward(self, features, offset):
        return self.last(self.norm(self.first(features)).tanh()) + offset


def objective(model, data):
    prediction = model(data["image"], data["context"]["offset"])
    error = prediction - data["target"]
    return error.square().sum(), {"error_sum": error.detach().sum()}


def split(rows):
    def microbatches(update):
        data = update["data"]
        for begin in range(0, data["image"].shape[0], rows):
            end = begin + rows
            yield (
                {
                    "image": data["image"][begin:end],
                    "context": {"offset": data["context"]["offset"][begin:end]},
                    "target": data["target"][begin:end],
                },
                update["scale"],
            )

    return microbatches


def test_shadowspill_generic_candidates_shared_forward_and_checkpoint(tmp_path):
    torch.manual_seed(42)
    model = Network()
    initial = copy.deepcopy(model.state_dict())
    reference = copy.deepcopy(model)
    batch = {
        "image": torch.randn(8, 6),
        "context": {"offset": torch.randn(8, 3)},
        "target": torch.randn(8, 3),
    }

    def groups(m):
        return [
            {"params": list(m.first.parameters()), "weight_decay": 0.01},
            {
                "params": [*m.norm.parameters(), *m.last.parameters()],
                "weight_decay": 0.0,
            },
        ]

    optimizer = torch.optim.AdamW(groups(reference), lr=0.002, betas=(0.8, 0.95))
    with (
        ShadowSpill(
            device="cuda:0",
            execution_gib=2,
            spill_gib=1,
            artifact_store=tmp_path / "artifacts",
            profiling_options=CORRECTNESS_PROFILING,
        ) as backend,
        Trainer(
            model,
            objective=objective,
            optimizer=torch.optim.AdamW,
            optimizer_args={"lr": 0.002},
            parameter_groups=groups,
            schedules={"betas": lambda _: (0.8, 0.95)},
            microbatches={"two_rows": split(2), "four_rows": split(4)},
            backend=backend,
        ) as trainer,
    ):
        trainer.prepare({"data": batch, "scale": 1 / 24})
        assert trainer.selected_candidate in {"two_rows", "four_rows"}
        trainer.planning.save(tmp_path / "search.json")
        prepared_state = trainer._execution.call.state_dict()["model"]
        for name, value in initial.items():
            torch.testing.assert_close(prepared_state[name], value)
        del prepared_state
        # A lower budget's admission may restore CPU-backed model views.
        # Compare state through checkpoint/export after calls, not raw views.
        with Forward(
            trainer.model,
            forward_fn=lambda m, x: {
                "prediction": m(x["image"], x["context"]["offset"])
            },
            backend=backend,
        ) as forward:
            forward.prepare(batch)
            for scale in (1 / 24, 1 / 48):
                result = trainer.step({"data": batch, "scale": scale})
                expected_loss, _ = objective(reference, batch)
                (expected_loss * scale).backward()
                optimizer.step()
                optimizer.zero_grad()
                assert abs(result.loss - float(expected_loss.detach() * scale)) < 2e-4
                output = forward(batch)
                expected = reference(batch["image"], batch["context"]["offset"])
                torch.testing.assert_close(
                    output["prediction"].cpu(), expected.detach(), rtol=2e-4, atol=2e-5
                )
                del output
        checkpoint = trainer.save(tmp_path / "checkpoint")
        saved = torch.load(
            checkpoint / "state.pt", map_location="cpu", weights_only=True
        )
        for name, weight in reference.state_dict().items():
            torch.testing.assert_close(
                saved["model"][name], weight, rtol=2e-4, atol=2e-5
            )
        assert trainer.step_count == 2
        trainer.load(checkpoint)
        assert trainer.step_count == 2


class CountingNetwork(Network):
    def __init__(self):
        super().__init__()
        self.register_buffer("counter", torch.tensor(0, dtype=torch.int64))

    def forward(self, features, offset):
        self.counter.add_(1)
        return super().forward(features, offset)


def test_mutable_buffer_survives_diagnostics_and_repeated_training(tmp_path):
    """Mutation outputs must clear the retained buffer's retired GPU binding."""
    from mlops.optim import AdamW

    torch.manual_seed(814)
    model = CountingNetwork()
    reference = copy.deepcopy(model)
    batch = {
        "image": torch.randn(4, 6),
        "context": {"offset": torch.randn(4, 3)},
        "target": torch.randn(4, 3),
    }
    optimizer = torch.optim.AdamW(reference.parameters(), lr=0.001, foreach=False)
    with (
        ShadowSpill(
            device="cuda:0",
            execution_gib=2,
            spill_gib=1,
            artifact_store=tmp_path / "artifacts",
            partition="whole",
            profiling_options=CORRECTNESS_PROFILING,
        ) as backend,
        Trainer(
            model,
            objective=objective,
            optimizer=AdamW,
            optimizer_args={"lr": 0.001, "opt_state_dtype": torch.float32},
            backend=backend,
        ) as trainer,
    ):
        trainer.prepare(batch)
        call = trainer._execution.call
        initial = call.state_dict()
        cpu_rng = torch.get_rng_state()
        device_rng = torch.cuda.get_rng_state()
        diagnostic = trainer.diagnose(batch, directory=tmp_path / "startup")
        assert diagnostic["buffer_snapshot_bytes"] == 8
        assert trainer.step_count == 0
        actual_leaves, actual_tree = torch.utils._pytree.tree_flatten(call.state_dict())
        initial_leaves, initial_tree = torch.utils._pytree.tree_flatten(initial)
        assert actual_tree == initial_tree
        for actual, before in zip(actual_leaves, initial_leaves, strict=True):
            if isinstance(actual, torch.Tensor):
                torch.testing.assert_close(actual, before, rtol=0, atol=0)
            else:
                assert actual == before
        assert torch.equal(torch.get_rng_state(), cpu_rng)
        assert torch.equal(torch.cuda.get_rng_state(), device_rng)
        assert (tmp_path / "startup/timelines/traced.html").is_file()
        for step in range(1, 4):
            result = trainer.step(batch)
            expected, _ = objective(reference, batch)
            expected.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            assert result.loss == pytest.approx(float(expected.detach()), rel=2e-5)
            state = call.state_dict()["model"]
            assert state["counter"].item() == step
            for name, value in reference.state_dict().items():
                torch.testing.assert_close(state[name], value, rtol=2e-4, atol=2e-5)


class RandomNetwork(Network):
    def forward(self, features, offset):
        return super().forward(features, offset) + 0.125 * torch.rand_like(offset)


def test_first_stochastic_update_is_independent_of_preparation_and_diagnostics(
    tmp_path,
):
    """Fresh/cached profiles and extra warmups must not change real random draws."""
    from dataclasses import replace
    from fractions import Fraction

    from mlops.optim import AdamW

    from shadowspill.planner import SearchOptions
    from shadowspill.planner.search.algorithms.pressurefit import (
        PressureFit,
        PressureFitOptions,
    )
    from tests.precision import low_precision_dtype

    dtype = low_precision_dtype()
    torch.manual_seed(1009)
    source = RandomNetwork().to(dtype=dtype)
    batch = {
        "image": torch.randn(4, 6, dtype=dtype),
        "context": {"offset": torch.randn(4, 3, dtype=dtype)},
        "target": torch.randn(4, 3, dtype=dtype),
    }
    results = []
    with ShadowSpill(
        device="cuda:0", execution_gib=2, spill_gib=1,
        artifact_store=tmp_path / "artifacts", partition="whole",
        search_options=SearchOptions(
            algorithm=PressureFit(
                PressureFitOptions(resolution_options=(Fraction(0),))
            )
        ),
    ) as backend:
        for ordinal, (profile_warmup, diagnostic_warmup) in enumerate(
            ((1, None), (7, 2), (1, None))
        ):
            backend.profiling_options = replace(
                CORRECTNESS_PROFILING, warmup_iterations=profile_warmup
            )
            with Trainer(
                copy.deepcopy(source), objective=objective, optimizer=AdamW,
                optimizer_args={
                    "lr": 0.001, "gradient_dtype": "parameter",
                    "opt_state_dtype": dtype, "parameter_rounding": "stochastic",
                    "opt_state_rounding": "stochastic",
                },
                backend=backend,
            ) as trainer:
                torch.manual_seed(1709)
                cpu_rng = torch.get_rng_state()
                device_rng = torch.cuda.get_rng_state(backend.device)
                trainer.prepare(batch)
                assert torch.equal(torch.get_rng_state(), cpu_rng)
                assert torch.equal(torch.cuda.get_rng_state(backend.device), device_rng)
                if diagnostic_warmup is not None:
                    trainer.diagnose(
                        batch, directory=tmp_path / f"diagnostic-{ordinal}",
                        warmup=diagnostic_warmup,
                    )
                    assert torch.equal(torch.get_rng_state(), cpu_rng)
                    assert torch.equal(
                        torch.cuda.get_rng_state(backend.device), device_rng
                    )
                first = trainer.step(batch)
                state = trainer._execution.call.state_dict()
                leaves, structure = torch.utils._pytree.tree_flatten(state)
                results.append((first.loss, leaves, structure))
    expected_loss, expected_leaves, expected_structure = results[0]
    for loss, leaves, structure in results[1:]:
        assert loss == expected_loss
        assert structure == expected_structure
        for actual, expected in zip(leaves, expected_leaves, strict=True):
            if isinstance(actual, torch.Tensor):
                assert torch.equal(
                    actual.reshape(-1).view(torch.uint8),
                    expected.reshape(-1).view(torch.uint8),
                )
            else:
                assert actual == expected
