"""Compute weights, stored gradients and optimizer masters retain distinct dtypes."""

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


def objective(model, x):
    result = model(x).float().sum()
    return result, {"sum": result.detach()}


@pytest.mark.cuda
@pytest.mark.parametrize("gradient_dtype", [None, torch.float32])
def test_mixed_precision_master_update_and_checkpoint(tmp_path, gradient_dtype):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    with torch.device("meta"):
        template = nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(template.parameters(), lr=0.001, foreach=False)
    examples = [
        ((torch.full((4, 8), value, dtype=torch.bfloat16),), {})
        for value in (0.125, 0.25)
    ]
    program = build_step(
        template,
        optimizer=optimizer,
        objective=objective,
        example_args=examples[0][0],
        num_examples=2,
        partition="whole",
        master_dtype=torch.float32,
        grad_dtype=gradient_dtype,
    )
    assert template.weight.dtype == torch.bfloat16 and not optimizer.state
    assert optimizer.param_groups[0]["params"][0] is template.weight
    bindings = {item["name"]: item for item in program.bindings["optimizer_bindings"]}
    assert bindings["weight"]["dtype"] == "torch.float32"
    assert bindings["compute.weight"]["dtype"] == "torch.bfloat16"
    assert bindings["gradient.weight"]["dtype"] == str(gradient_dtype or torch.bfloat16)
    assert bindings["optimizer.weight.exp_avg"]["dtype"] == "torch.float32"
    profiles = measure_program(program, warmup_seconds=0.001, measure_seconds=0.002)
    plan = plan_program(
        apply_profiles(program, profiles),
        memory_budgets=MemoryBudgets(2 << 30, 64 << 20),
        communication_info=CommunicationInfo(1e10, 1e10),
    )
    plan.save(tmp_path / "plan.ssp")
    reference = nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        reference.weight.copy_(torch.arange(64).reshape(8, 8) / 64)
    master = nn.Parameter(reference.weight.float().detach().clone())
    opt = torch.optim.AdamW([master], lr=0.001, foreach=False)
    initial = {"weight": reference.weight.detach().clone()}
    expected, parameter_metrics = [], []
    for index in range(3):
        reference.zero_grad(set_to_none=True)
        outputs = []
        for args, kwargs in examples:
            loss, metrics = objective(reference, *args, **kwargs)
            loss.backward()
            outputs.append((loss.detach(), metrics))
        master.grad = reference.weight.grad.float()
        parameter_metrics.append(
            {
                "weight": {
                    "param_norm": torch.linalg.vector_norm(
                        reference.weight.detach(), dtype=torch.float32
                    ),
                    "grad_norm": torch.linalg.vector_norm(
                        reference.weight.grad, dtype=torch.float32
                    ),
                }
            }
            if index != 1
            else None
        )
        opt.step()
        with torch.no_grad():
            reference.weight.copy_(master)
        state = {
            "weight": reference.weight.detach().clone(),
            "master.weight": master.detach().clone(),
            **{
                f"optimizer.weight.{name}": value.clone()
                for name, value in opt.state[master].items()
            },
        }
        expected.append((outputs, state))
    torch.save(
        dict(
            initial=initial,
            examples=examples,
            expected=expected,
            hyperparams=[{}, {}, {}],
            accumulate=[False] * 3,
            restore=False,
            parameter_metrics=parameter_metrics,
        ),
        tmp_path / "oracle.pt",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(Path(__file__).with_name("_execute.py")),
            str(tmp_path),
        ],
        env=os.environ,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
