"""A persisted preparation trains correctly in fresh processes at two budgets."""

from __future__ import annotations

import copy
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("custom_ops", [False, True])
def test_prepared_training_survives_process_exit_and_changes_budget(
    tmp_path, custom_ops
):
    # Each process installs its own allocator before touching CUDA. The second
    # and third workers must reuse the first worker's objective and selection.
    for mode, budget in (("prepare", 3), ("train", 2), ("train", 3)):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "tests.training.test_prepared_processes",
                mode,
                str(tmp_path),
                str(budget),
                str(int(custom_ops)),
            ],
            text=True,
            capture_output=True,
            timeout=240,
        )
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prepared_llama_rebinds_rotary_buffers_in_fresh_processes(tmp_path):
    for mode, budget in (("llama-prepare", 3), ("llama-train", 2), ("llama-train", 3)):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "tests.training.test_prepared_processes",
                mode,
                str(tmp_path),
                str(budget),
            ],
            text=True,
            capture_output=True,
            timeout=240,
        )
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_managed_data_parallel_opens_groups_after_runtime_and_trains(tmp_path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.training.test_prepared_processes",
            "parallel",
            str(tmp_path),
            "2",
        ],
        env={
            **os.environ,
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "RANK": "0",
            "WORLD_SIZE": "1",
            "LOCAL_RANK": "0",
        },
        text=True,
        capture_output=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_trainer_partial_checkpoint_streams_and_restores_without_the_frozen_base(
    tmp_path,
):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.training.test_prepared_processes",
            "checkpoint",
            str(tmp_path),
            "2",
        ],
        text=True,
        capture_output=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _objective(model, batch):
    features, target = batch
    return (model(features) - target).square().sum()


def _split(batch):
    features, target = batch
    for begin in range(0, len(features), 2):
        yield (
            (features[begin : begin + 2].clone(), target[begin : begin + 2].clone()),
            1 / target.numel(),
        )


class _LazyEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(8, 4))
        self.head = nn.Linear(4, 2)
        self.register_buffer("offset", torch.randn(4), persistent=False)
        self.custom_ops = True

    def forward(self, tokens):
        if self.custom_ops:
            import mlops

            hidden = mlops.embedding(tokens, self.weight)
        else:
            hidden = torch.nn.functional.embedding(tokens, self.weight)
        return self.head(hidden + self.offset)


def _worker(mode: str, directory: Path, budget: int, custom_ops: bool = False) -> None:
    if mode.startswith("llama-"):
        _llama_worker(mode, directory, budget)
        return
    if mode == "checkpoint":
        _checkpoint_worker(directory)
        return
    from unittest.mock import patch

    from qualification.profiling import CORRECTNESS_PROFILING
    from shadowspill.pytorch import read_model_state
    from shadowspill.training import DataParallel, PreparedTraining, Trainer
    from shadowspill.training.backends import ShadowSpill

    torch.manual_seed(720)
    model = (
        _LazyEmbedding()
        if custom_ops
        else nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 2))
    )
    reference = copy.deepcopy(model)
    if custom_ops:
        reference.custom_ops = False
    batch = (
        torch.randint(0, 8, (8,)) if custom_ops else torch.randn(8, 4),
        torch.randn(8, 2),
    )
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    backend = ShadowSpill(
        device="cuda:0",
        execution_gib=budget,
        spill_gib=1,
        artifact_store=directory / "store",
        partition="whole",
        # Collective phase instrumentation compares independent timing windows;
        # retain normal clock conditioning for that timing-sensitive path.
        profiling_options=None if mode == "parallel" else CORRECTNESS_PROFILING,
    )
    with Trainer(
        model,
        objective=_objective,
        optimizer=torch.optim.SGD,
        optimizer_args={"lr": 0.01},
        microbatch_candidates={"two_rows": _split},
        backend=backend,
        distributed=DataParallel(shard_weights=True, shard_gradients=True)
        if mode == "parallel"
        else None,
    ) as trainer:
        if mode == "prepare":
            prepared = trainer.prepare_artifacts(
                batch, budgets=[(2 << 30, 1 << 30), (3 << 30, 1 << 30)]
            )
            assert len(prepared.candidates["two_rows"].exports) == 1
            if custom_ops:
                assert "mlops.providers.builtin.embedding" in (
                    prepared.candidates["two_rows"].operator_modules
                )
            assert len(prepared.search.winner_plans) == 2
            pool = backend.runtime.pools["execution"]
            overhead = pool.physical_capacity - pool.capacity
            assert overhead > 0
            for program in prepared.candidates["two_rows"].programs:
                assert program.problem.source_execution_budget_bytes == budget << 30
                assert program.problem.fixed_execution_bytes >= overhead
            for (execution, _), plan in prepared.search.winner_plans.items():
                assert (
                    overhead < plan.summary.physical_execution_peak_bytes <= execution
                )
            assert trainer.step_count == 0
            prepared.save(directory / "prepared")
            return

        if mode == "parallel":
            trainer.prepare(batch)
        else:
            if custom_ops:
                assert "mlops.providers.builtin.embedding" not in sys.modules
            prepared = PreparedTraining.load(
                directory / "prepared", budget=backend.budget
            )
            assert len(prepared.search.winner_plans) == 1
            assert len(prepared.candidates["two_rows"].programs) == 1
            with (
                patch(
                    "shadowspill.pytorch.planning.training.capture._capture_training_objectives",
                    side_effect=AssertionError(
                        "admission exported the objective again"
                    ),
                ),
                patch(
                    "shadowspill.pytorch.planning.training.build.plan_training_programs",
                    side_effect=AssertionError("admission searched for another plan"),
                ),
            ):
                trainer.prepare(batch, prepared=prepared)
        # Two updates exercise both the first gradient creation and subsequent
        # accumulation, including resetting the gradient at the next update.
        for _ in range(2):
            result = trainer.step(batch)
            optimizer.zero_grad(set_to_none=True)
            expected = _objective(reference, batch) / batch[1].numel()
            expected.backward()
            optimizer.step()
            assert result.loss == pytest.approx(float(expected.detach()), rel=2e-5)
            actual = read_model_state(trainer.model, runtime=backend.runtime)
            for name, value in reference.state_dict().items():
                torch.testing.assert_close(actual[name], value, rtol=2e-4, atol=2e-5)
    assert not backend.is_open


def _llama_objective(model, batch):
    tokens, targets = batch
    return model.loss(
        tokens.reshape(1, -1),
        targets.reshape(1, -1),
        seq_lens=(tokens.shape[1],) * tokens.shape[0],
        reduction="sum",
    )


def _llama_worker(mode, directory, budget):
    from unittest.mock import patch

    from qualification.profiling import CORRECTNESS_PROFILING
    from shadowspill.planner import StepDataOrdering
    from shadowspill.pytorch import read_model_state
    from shadowspill.training import PreparedTraining, Trainer
    from shadowspill.training.backends import ShadowSpill
    from workloads.models.llama3.config import Config
    from workloads.models.llama3.implementations.mlops import Model
    from workloads.models.llama3.implementations.pytorch import Model as Reference

    torch.manual_seed(720)
    config = Config(2, 128, 4, 2, 256, 128, max_seq_len=32)
    model = Model(config)
    reference = Reference(config)
    reference.load_state_dict(model.state_dict())
    batch = (torch.randint(0, 128, (4, 16)), torch.randint(0, 128, (4, 16)))
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    backend = ShadowSpill(
        execution_gib=budget,
        spill_gib=1,
        artifact_store=directory / "store",
        orderings=[StepDataOrdering(1, 2), StepDataOrdering(2, 1)],
        profiling_options=CORRECTNESS_PROFILING,
    )
    trainer = Trainer(
        model,
        objective=_llama_objective,
        optimizer=torch.optim.SGD,
        optimizer_args={"lr": 0.01},
        microbatch_candidates={"two_sequences": _split},
        backend=backend,
    )
    if mode == "llama-prepare":
        prepared = trainer.prepare_artifacts(
            batch, budgets=[(2 << 30, 1 << 30), (3 << 30, 1 << 30)]
        )
        assert len(prepared.candidates["two_sequences"].exports) == 1
        prepared.save(directory / "prepared")
    else:
        assert "mlops.providers.builtin.rope" not in sys.modules
        prepared = PreparedTraining.load(directory / "prepared", budget=backend.budget)
        with (
            patch(
                "shadowspill.pytorch.planning.training.capture._capture_training_objectives",
                side_effect=AssertionError("admission exported the objective again"),
            ),
            patch(
                "shadowspill.pytorch.planning.training.build.plan_training_programs",
                side_effect=AssertionError("admission searched for another plan"),
            ),
        ):
            trainer.prepare(batch, prepared=prepared)
        for _ in range(2):
            result = trainer.step(batch)
            optimizer.zero_grad(set_to_none=True)
            expected = _llama_objective(reference, batch) / batch[1].numel()
            expected.backward()
            optimizer.step()
            assert result.loss == pytest.approx(float(expected.detach()), rel=2e-5)
            actual = read_model_state(trainer.model, runtime=backend.runtime)
            for name, value in reference.state_dict().items():
                torch.testing.assert_close(actual[name], value, rtol=2e-4, atol=2e-5)
    trainer.close()


def _checkpoint_worker(directory):
    from qualification.profiling import CORRECTNESS_PROFILING
    from shadowspill.pytorch import read_model_state
    from shadowspill.training import Trainer
    from shadowspill.training.backends import ShadowSpill
    from tests.training.test_training_checkpoints import PartlyTrained

    batch = torch.ones(2, 3, dtype=torch.bfloat16)
    initial = PartlyTrained().state_dict()
    with ShadowSpill(
        execution_gib=2,
        spill_gib=1,
        partition="whole",
        artifact_store=directory / "store",
        profiling_options=CORRECTNESS_PROFILING,
    ) as backend:

        def make(model):
            return Trainer(
                model,
                objective=lambda m, x: m(x).square().mean(),
                optimizer=torch.optim.AdamW,
                optimizer_args={"lr": 0.01, "foreach": False},
                master_dtype=torch.float32,
                grad_dtype=torch.float32,
                frozen_state_id="base-v1",
                backend=backend,
            )

        first = make(PartlyTrained()).prepare(batch)
        first.step(batch)
        checkpoint = first.save(directory / "saved")
        saved = torch.load(checkpoint / "state.pt", mmap=True, weights_only=True)
        assert set(saved["model"]) == {"delta", "calls", "moving"}
        assert saved["model"]["delta"].dtype == torch.float32
        assert len(saved["optimizer"]["state"]) == 1
        parameter_checkpoints = {
            weights: first.save(
                directory / weights, weights=weights, include_optimizer=False
            )
            for weights in ("master", "compute")
        }
        first.step(batch)
        expected = read_model_state(first.model, runtime=backend.runtime)
        first.close()

        resumed = make(PartlyTrained(device="meta"))
        resumed.prepare(
            batch,
            initialize=lambda m: m.load_state_dict(initial),
            checkpoint=checkpoint,
        )
        resumed.step(batch)
        actual = read_model_state(resumed.model, runtime=backend.runtime)
        for name, value in expected.items():
            torch.testing.assert_close(actual[name], value, rtol=0, atol=0)
        resumed.load(checkpoint)
        resumed.step(batch)
        actual = read_model_state(resumed.model, runtime=backend.runtime)
        for name, value in expected.items():
            torch.testing.assert_close(actual[name], value, rtol=0, atol=0)
        resumed.close()

        for weights, path in parameter_checkpoints.items():
            payload = torch.load(path / "state.pt", mmap=True, weights_only=True)
            assert "optimizer" not in payload
            assert set(payload["model"]) == {"delta", "calls", "moving"}
            assert payload["model"]["delta"].dtype == (
                torch.float32 if weights == "master" else torch.bfloat16
            )
            # Resume progress, but compare the update against a freshly constructed
            # eager optimizer using the checkpoint's sole parameter representation.
            from shadowspill.training.backends import PyTorch

            reference = Trainer(
                PartlyTrained(),
                objective=lambda m, x: m(x).square().mean(),
                optimizer=torch.optim.AdamW,
                optimizer_args={"lr": 0.01, "foreach": False},
                master_dtype=torch.float32,
                grad_dtype=torch.float32,
                frozen_state_id="base-v1",
                backend=PyTorch(device="cpu", compile=False),
            ).prepare(batch, checkpoint=path)
            reference.step(batch)
            resumed = make(PartlyTrained()).prepare(batch, checkpoint=path)
            assert resumed.step_count == 1
            resumed.step(batch)
            actual = read_model_state(resumed.model, runtime=backend.runtime)
            for name, value in reference.model.state_dict().items():
                torch.testing.assert_close(actual[name], value, rtol=2e-3, atol=2e-4)
            with pytest.raises(ValueError, match="fresh callable"):
                resumed.load(path)
            resumed.close()
            reference.close()


if __name__ == "__main__":
    _worker(
        sys.argv[1],
        Path(sys.argv[2]),
        int(sys.argv[3]),
        len(sys.argv) > 4 and bool(int(sys.argv[4])),
    )
