"""Frozen-base checkpoints replay a saved training/evaluation pair in cold workers."""

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
    build_forward,
    build_step,
    measure_program,
    plan_program,
)


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Parameter(
            torch.full((3, 2), 0.125, dtype=torch.bfloat16), requires_grad=False
        )
        self.tied_base = self.base
        self.delta = nn.Parameter(torch.zeros(3, 2, dtype=torch.bfloat16))
        self.moving = nn.Parameter(torch.tensor(0.01), requires_grad=False)
        self.evaluation_only = nn.Parameter(torch.tensor(0.02), requires_grad=False)
        self.register_buffer("calls", torch.tensor(0.0))

    def forward(self, x):
        with torch.no_grad():
            self.moving.add_(0.001)
            self.calls.add_(1)
        return self.predict(x)

    def predict(self, x):
        return (
            (x @ (self.base + self.delta)).float()
            + self.moving
            + self.calls
            + self.evaluation_only
        )


def objective(model, x):
    return model(x).square().mean()


def evaluation(model, x):
    return model.predict(x).mean()


@pytest.mark.cuda
def test_partial_trainer_preserves_masters_mutations_evaluation_and_resume(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.save(Network().state_dict(), tmp_path / "initial.pt")
    inputs = (torch.ones(2, 3, dtype=torch.bfloat16),)
    with torch.device("meta"):
        model = Network()
    training = build_step(
        model,
        optimizer=torch.optim.AdamW(
            model.parameters(), lr=torch.tensor(0.01), foreach=False
        ),
        objective=objective,
        example_args=inputs,
        partition="whole",
        master_dtype=torch.float32,
    )
    forward = build_forward(
        model, forward_fn=evaluation, example_args=inputs, partition="whole"
    )
    for name, program in (("training", training), ("evaluation", forward)):
        profiles = measure_program(program, warmup_seconds=0.001, measure_seconds=0.002)
        plan = plan_program(
            apply_profiles(program, profiles),
            memory_budgets=MemoryBudgets(2 << 30, 64 << 20),
            communication_info=CommunicationInfo(1e10, 1e10),
        )
        plan.save(tmp_path / f"{name}.ssp")
    for mode in ("whole", "resume"):
        worker = subprocess.run(
            [
                sys.executable,
                "-S",
                str(Path(__file__).with_name("_partial_train.py")),
                str(tmp_path),
                mode,
            ],
            env=os.environ,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert worker.returncode == 0, worker.stdout + worker.stderr
        assert f"Qualified partial Trainer {mode}" in worker.stdout
