"""Prepared operation families share gradients, optimizer moments and counters."""

import inspect
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
    build_forward_backward,
    build_step,
    measure_program,
    plan_program,
)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(4, 4))
        self.tied = self.weight
        self.bias = nn.Parameter(torch.empty(4))

    def forward(self, x):
        return x @ self.weight + self.bias


def objective(model, x, target, mask):
    return ((model(x) - target).square() * mask).sum()


adam_update = inspect.unwrap(torch.optim.AdamW.step)


class CountedAdamW(torch.optim.AdamW):
    """An ordinary optimizer with an additional integral host counter."""

    @torch.no_grad()
    def step(self, closure=None):
        # Invoke the tensor update without torch.optim's eager graph-break wrapper,
        # as the capture frontend does for an optimizer's outermost step.
        result = adam_update(self, closure)
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    state = self.state[parameter]
                    if "updates" not in state:
                        state["updates"] = torch.zeros(
                            (), dtype=torch.int64, device="cpu"
                        )
                    state["updates"].add_(1)
        return result


@pytest.mark.cuda
def test_mutable_operation_families_in_cold_worker(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    examples = [
        (
            torch.arange(n * 4).reshape(n, 4).float() / 32,
            torch.full((n, 4), 0.125),
            torch.ones(n, 4),
        )
        for n in (2, 3)
    ]
    examples[1][2][0].zero_()
    with torch.device("meta"):
        model = Model()
    programs = {
        "evaluation": build_forward(
            model.eval(), example_args=(examples[0][0],), partition="whole"
        )
    }
    model.train()
    for index, args in enumerate(examples):
        programs[f"backward{index}"] = build_forward_backward(
            model, objective=objective, example_args=args, partition="whole"
        )
        programs[f"step{index}"] = build_step(
            model,
            optimizer=CountedAdamW(model.parameters(), lr=0.01, foreach=False),
            objective=objective,
            example_args=args,
            partition="whole",
        )
    for name, program in programs.items():
        profile = measure_program(program, warmup_seconds=0.001, measure_seconds=0.002)
        plan = plan_program(
            apply_profiles(program, profile),
            memory_budgets=MemoryBudgets(2 << 30, 64 << 20),
            communication_info=CommunicationInfo(1e10, 1e10),
        )
        plan.save(tmp_path / f"{name}.ssp")
    reference = Model()
    with torch.no_grad():
        reference.weight.copy_(torch.arange(16).reshape(4, 4) / 32)
        reference.bias.fill_(0.125)
    initial = {name: value.clone() for name, value in reference.state_dict().items()}
    gradients = []
    for args in examples:
        objective(reference, *args).backward()
        gradients.append(
            {name: p.grad.clone() for name, p in reference.named_parameters()}
        )
    optimizer = CountedAdamW(reference.parameters(), lr=0.01, foreach=False)
    updated = []
    for args in examples:
        optimizer.zero_grad(set_to_none=True)
        objective(reference, *args).backward()
        optimizer.step()
        state = {name: p.detach().clone() for name, p in reference.named_parameters()}
        state.update(
            {
                f"optimizer.{name}.{key}": value.clone()
                for name, p in reference.named_parameters()
                for key, value in optimizer.state[p].items()
            }
        )
        updated.append(state)
    torch.save(
        dict(initial=initial, examples=examples, gradients=gradients, updated=updated),
        tmp_path / "oracle.pt",
    )
    worker = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("_mutable_shared.py")),
            str(tmp_path),
        ],
        env={
            **os.environ,
            "TORCHINDUCTOR_CACHE_DIR": str(tmp_path / "cold-inductor"),
            "TRITON_CACHE_DIR": str(tmp_path / "cold-triton"),
        },
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert worker.returncode == 0, worker.stdout + worker.stderr
    assert "Qualified mutable shared Plans" in worker.stdout
