"""Unequal nested examples and runtime controls match an unsplit Adam update."""

import copy
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from shadowspill import (
    CommunicationInfo,
    MemoryBudgets,
    apply_profiles,
    build_step,
    measure_program,
    plan_program,
)
from tests.shadowspill.execution._oracle import optimizer_values


def objective(model, batch, *, scale):
    error = model(batch["features"]) - batch["target"]
    return error.square().mean() * scale, {"error_sum": error.detach().sum()}


def optimizer(model):
    return torch.optim.AdamW(
        [
            {
                "params": [parameter],
                "weight_decay": decay,
                "lr": torch.tensor(0.02),
                "betas": (torch.tensor(0.9), torch.tensor(0.99)),
            }
            for parameter, decay in ((model.weight, 0.25), (model.bias, 0.0))
        ],
        foreach=False,
    )


@pytest.mark.cuda
def test_mixed_calls_keep_weights_metrics_and_group_controls(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.manual_seed(10)
    model = nn.Linear(3, 2)
    initial = copy.deepcopy(model.state_dict())
    update = optimizer(model)
    examples, expected, controls = [], [], []
    for step, scale in enumerate((1.0, 0.5)):
        batch = {"features": torch.randn(5, 3), "target": torch.randn(5, 2)}
        calls = [
            (
                ({name: value[start:stop].clone() for name, value in batch.items()},),
                {"scale": torch.tensor(scale * (stop - start) / 5)},
            )
            for start, stop in ((0, 2), (2, 5))
        ]
        settings = {}
        for index, group in enumerate(update.param_groups):
            for field, value in (
                ("lr", 0.02 / (step + 1)),
                ("betas.0", 0.8 + 0.05 * index),
            ):
                target = group["lr"] if field == "lr" else group["betas"][0]
                target.fill_(value)
                settings[f"optimizer_group.{index}.{field}"] = value
        outputs = [objective(model, *args, **kwargs) for args, kwargs in calls]
        update.zero_grad(set_to_none=True)
        loss, _ = objective(model, batch, scale=scale)
        loss.backward()
        norms = {
            name: {
                "param_norm": parameter.detach().norm(),
                "grad_norm": parameter.grad.norm(),
            }
            for name, parameter in model.named_parameters()
        }
        update.step()
        examples.append(calls)
        controls.append(settings)
        expected.append(
            {
                "loss": loss.detach().clone(),
                "metrics": [metrics for _, metrics in outputs],
                "norms": norms,
                "values": {
                    **copy.deepcopy(model.state_dict()),
                    **optimizer_values(model, update),
                },
            }
        )
    torch.save(
        dict(initial=initial, examples=examples, controls=controls, expected=expected),
        tmp_path / "oracle.pt",
    )
    with torch.device("meta"):
        template = nn.Linear(3, 2)
    program = build_step(
        template,
        objective=objective,
        optimizer=optimizer(template),
        example_calls=examples[0],
        partition="whole",
    )
    assert all(parameter.is_meta for parameter in template.parameters())
    profiles = measure_program(program, warmup_seconds=0.001, measure_seconds=0.002)
    plan = plan_program(
        apply_profiles(program, profiles),
        memory_budgets=MemoryBudgets(2 << 30, 64 << 20),
        communication_info=CommunicationInfo(1e10, 1e10),
    )
    plan.save(tmp_path / "plan.ssp")
    child = subprocess.run(
        [
            sys.executable,
            "-S",
            str(Path(__file__).with_name("_mixed_inputs.py")),
            str(tmp_path),
        ],
        env=os.environ,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    assert "Qualified mixed-input Trainer" in child.stdout
