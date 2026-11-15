"""Supplied catalog training uses the same service as external definitions."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from shadowspill.artifacts.tensors import TensorChunk
from shadowspill_workloads.recipes.regression.initialize import fill_tensor
from shadowspill_workloads.recipes.regression.model import Regression
from workload_tests._regression_service_client import (
    DEFAULTS,
    INPUTS,
    TARGETS,
    VALID,
    WEIGHTS,
    Oracle,
)


def initialized():
    model = Regression()
    with torch.no_grad():
        for name, value in model.state_dict().items():
            fill_tensor(
                TensorChunk(name, tuple(value.shape), value.dtype, 0, value.numel()),
                value.reshape(-1),
            )
    return model


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_independent_regression_adamw_oracle_matches_cpu_autograd(dtype):
    model = initialized()
    oracle = Oracle(
        {name: value.detach().numpy() for name, value in model.named_parameters()},
        dtype,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=DEFAULTS["lr"], foreach=False)
    for overrides, scale in (
        (
            {
                "lr": 0.003,
                "beta1": 0.8,
                "beta2": 0.95,
                "eps": 1e-6,
                "weight_decay": 0.02,
            },
            0.5,
        ),
        ({}, 1.0),
    ):
        settings = {
            name: float(np.float32(value))
            for name, value in {**DEFAULTS, **overrides}.items()
        }
        group = optimizer.param_groups[0]
        for name in ("lr", "eps", "weight_decay"):
            group[name] = settings[name]
        group["betas"] = (settings["beta1"], settings["beta2"])
        prediction = model(torch.from_numpy(INPUTS))
        loss = (
            (prediction - torch.from_numpy(TARGETS)).square().sum(dim=-1)
            * torch.from_numpy(WEIGHTS * VALID)
        ).sum()
        loss.backward()
        predicted, expected_loss, gradients = oracle.forward_backward()
        np.testing.assert_allclose(
            prediction.detach().numpy(), predicted, rtol=2e-5, atol=1e-6
        )
        np.testing.assert_allclose(loss.item(), expected_loss, rtol=2e-5, atol=1e-6)
        for name, value in model.named_parameters():
            np.testing.assert_allclose(
                value.grad.numpy(), gradients[name], rtol=2e-5, atol=1e-6
            )
            value.grad.mul_(scale)
            if dtype == "bfloat16":
                value.grad = value.grad.to(torch.bfloat16).float()
        optimizer.step()
        optimizer.zero_grad()
        oracle.update(gradients, overrides, scale)
        for name, value in model.named_parameters():
            np.testing.assert_allclose(
                value.detach().numpy(), oracle.weights[name], rtol=2e-5, atol=1e-6
            )


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_supplied_regression_public_step_split_and_adamw_resume(tmp_path, dtype):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    model = initialized()
    (tmp_path / "setup.json").write_text(
        json.dumps(
            {
                "initial": {
                    name: value.detach().tolist()
                    for name, value in model.named_parameters()
                },
                "gradient_dtype": dtype,
            }
        )
    )
    server = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("_regression_service_server.py")),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=720,
    )
    (tmp_path / "server.stdout").write_text(server.stdout)
    (tmp_path / "server.stderr").write_text(server.stderr)
    assert server.returncode == 0, server.stdout + server.stderr
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert (
        receipt["all_client_claims_released"] and receipt["all_server_claims_released"]
    )
    assert receipt["whole_split_equivalence"] and receipt["evaluation_isolation"]
    assert receipt["adamw_resume"] and receipt["zero_lr_preserves_moments"]
    assert len(receipt["runs"]) == 2
