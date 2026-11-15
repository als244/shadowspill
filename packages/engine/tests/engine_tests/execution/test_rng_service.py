"""Stochastic two-layer model through a public service and an independent client."""

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from engine_tests.execution._rng_service_client import PIXELS, WEIGHTS, Oracle
from engine_tests.fixtures import stochastic_definition
from engine_tests.fixtures.definition_operations import example
from engine_tests.fixtures.installed_definition import install
from engine_tests.training.test_setup_resolution import arguments
from shadowspill.artifacts.tensors import TensorChunk
from shadowspill.pytorch._rng import preserve_rng
from shadowspill.pytorch.capture.objective import capture_training_objective
from shadowspill.training.definitions import InitializationRuntime, load_registry
from shadowspill.training.preparation.setup import resolve_training
from shadowspill.training.worker.initialization import prepare_initialization
from shadowspill_client import (
    BuildCandidates,
    Execution,
    Memory,
    PlanningOptions,
    Preparation,
    ProfilingOptions,
)


@pytest.mark.parametrize("frozen", [False, True])
def test_independent_numpy_oracle_matches_autograd_for_weighted_split_updates(frozen):
    with preserve_rng(torch.device("cpu")):
        torch.manual_seed(203)
        model = stochastic_definition.create(SimpleNamespace(num_classes=3))
        if frozen:
            for value in model.encoder.parameters():
                value.requires_grad_(False)
        reference = Oracle(
            {name: value.detach().numpy() for name, value in model.named_parameters()},
            trainable=["head.*"] if frozen else None,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.02, foreach=False)
        for temperature in (0.5, 1.0):
            prediction, noise = model(torch.from_numpy(PIXELS.reshape(3, 1, 2, 2)))
            loss = (
                torch.nn.functional.cross_entropy(
                    prediction / temperature, torch.arange(3), reduction="none"
                )
                * torch.from_numpy(WEIGHTS)
            ).sum()
            loss.backward()
            reference.check(
                {
                    "prediction": prediction.detach().numpy(),
                    "noise": noise.detach().numpy(),
                },
                SimpleNamespace(loss=loss.item()),
                temperature=temperature,
                backward=True,
            )
        for name, value in model.named_parameters():
            if not value.requires_grad:
                continue
            np.testing.assert_allclose(
                reference.gradients[name], value.grad.numpy(), rtol=1e-5, atol=1e-6
            )
            value.grad.mul_(-0.5)
        optimizer.step()
        reference.update(0.02, -0.5)
        for name, value in model.named_parameters():
            np.testing.assert_allclose(
                reference.weights[name], value.detach().numpy(), rtol=1e-6, atol=1e-7
            )


def configured(tmp_path, monkeypatch, *, trainable=None):
    installed = install(tmp_path / "site-packages", stochastic=True)
    for name in list(sys.modules):
        if name.split(".")[0] == "external_classifier_fixture":
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(str(installed))
    monkeypatch.setenv(
        "PYTHONPATH", str(installed) + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    return resolve_training(
        load_registry(["external-classifier-fixture"]),
        **{
            **arguments(),
            "model": replace(arguments()["model"], trainable=trainable),
            "memory": Memory(device_gib=2, spill_gib=0.0625),
        },
        execution=Execution(seed=37),
        operations={
            "step": ["metrics"],
            "forward_backward": ["metrics"],
            "forward": ["predictions"],
        },
        prepare=Preparation(
            candidates=BuildCandidates(microbatch_sizes=[2, 4]),
            planning=PlanningOptions(microbatch_counts=[1, 2]),
            profiling=ProfilingOptions(
                conditioning_seconds=0.001, measurement_seconds=0.002
            ),
        ),
    )


def test_installed_stochastic_capture_preserves_padding_and_real_gradients(
    tmp_path, monkeypatch
):
    setup = configured(tmp_path, monkeypatch)
    with (
        preserve_rng(torch.device("cpu")),
        example(setup.model, setup.objective, "forward_backward") as (binding, adapted),
    ):
        model = setup.model.create()
        args = binding.arguments(adapted, {"temperature": 0.5})
        captured = capture_training_objective(model, binding.objective, args)
        state = torch.get_rng_state()
        loss, noise, prediction, valid = captured.exported.exported_program.module()(
            *args
        )
        torch.set_rng_state(state)
        expected = binding.objective(model, *args)
        torch.testing.assert_close(loss, expected.loss, rtol=0, atol=0)
        assert (
            noise.shape == (4, 8) and prediction.shape == (4, 3) and valid.item() == 3
        )
        actual_grad = torch.autograd.grad(
            loss, tuple(model.parameters()), retain_graph=True
        )
        expected_grad = torch.autograd.grad(expected.loss, tuple(model.parameters()))
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


@pytest.mark.cuda
def test_public_stochastic_step_split_and_checkpoint_resume(tmp_path, monkeypatch):
    setup = configured(tmp_path, monkeypatch)
    model = setup.model.create()
    with (
        prepare_initialization(setup)[0].description.open(
            InitializationRuntime(
                tmp_path, lambda _: pytest.fail("unexpected weight artifact")
            )
        ) as fill,
        torch.no_grad(),
    ):
        for name, value in model.named_parameters():
            fill(
                TensorChunk(name, tuple(value.shape), value.dtype, 0, value.numel()),
                value.reshape(-1),
            )
    (tmp_path / "setup.json").write_text(
        json.dumps(
            {
                "config": setup.config.to_dict(),
                "initial": {
                    name: value.detach().tolist()
                    for name, value in model.named_parameters()
                },
            }
        )
    )
    server = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("_rng_service_server.py")),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=660,
    )
    (tmp_path / "server.stdout").write_text(server.stdout)
    (tmp_path / "server.stderr").write_text(server.stderr)
    assert server.returncode == 0, server.stdout + server.stderr
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert (
        receipt["all_client_claims_released"] and receipt["all_server_claims_released"]
    )
    assert len(receipt["runs"]) == 3
    assert receipt["optional_trace_failures_injected"] == 1
    assert receipt["runs"][0]["trace_failure_preserves_update"]
    assert receipt["runs"][0]["trace_download_after_close"]
    assert all(len(run["runtime_traces"]) == 5 for run in receipt["runs"])
