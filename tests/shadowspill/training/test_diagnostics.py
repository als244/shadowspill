"""Saved startup diagnostics preserve training state and the first real update."""

import copy
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from shadowspill import (
    CommunicationInfo,
    MemoryBudgets,
    Program,
    apply_profiles,
    build_step,
    measure_program,
    plan_program,
)
from tests.shadowspill.execution._oracle import optimizer_values


class Network(nn.Module):
    def __init__(self, stochastic=False):
        super().__init__()
        self.first = nn.Linear(6, 8)
        self.norm = nn.LayerNorm(8)
        self.last = nn.Linear(8, 3)
        self.stochastic = stochastic
        self.register_buffer("counter", torch.tensor(0, dtype=torch.int64))

    def forward(self, features):
        self.counter.add_(1)
        value = self.last(self.norm(self.first(features)).tanh())
        return value + 0.125 * torch.rand_like(value) if self.stochastic else value


def objective(model, features, target):
    return (model(features) - target).square().mean()


def test_diagnostics_require_explicit_optimizer_guarantee_and_host_rates():
    from shadowspill.training.diagnostics.state import learning_rates

    bindings = {}
    path = SimpleNamespace(
        plan=SimpleNamespace(program=SimpleNamespace(bindings=bindings)),
        controls=SimpleNamespace(host={}, bindings={}),
    )
    with pytest.raises(ValueError, match="zero_lr_preserves_state=True"):
        learning_rates(path)
    bindings.update(
        optimizer_zero_lr_preserves_state=True, optimizer_learning_rates=["lr0", "lr1"]
    )
    path.controls.host["lr0"] = torch.tensor(0.1)
    path.controls.bindings["lr0"] = {"role": "hyperparameter"}
    with pytest.raises(ValueError, match="host-tensor learning rates"):
        learning_rates(path)
    path.controls.host["lr1"] = torch.tensor(0.1)
    path.controls.bindings["lr1"] = {"role": "hyperparameter"}
    assert learning_rates(path) == ("lr0", "lr1")


@pytest.mark.cuda
@pytest.mark.parametrize("stochastic", [False, True])
def test_saved_startup_diagnostics_preserve_state_and_rng(tmp_path, stochastic):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from mlops.optim import AdamW

    torch.manual_seed(814)
    dtype = torch.bfloat16 if stochastic else torch.float32
    model = Network(stochastic).to(dtype=dtype)
    args = (torch.randn(4, 6, dtype=dtype), torch.randn(4, 3, dtype=dtype))
    initial = copy.deepcopy(model.state_dict())
    expected = []
    if not stochastic:
        reference = copy.deepcopy(model)
        update = torch.optim.AdamW(reference.parameters(), lr=0.001, foreach=False)
        for _ in range(3):
            update.zero_grad(set_to_none=True)
            loss = objective(reference, *args)
            loss.backward()
            update.step()
            expected.append(
                {
                    "loss": float(loss.detach()),
                    "values": {
                        **copy.deepcopy(reference.state_dict()),
                        **optimizer_values(reference, update),
                    },
                }
            )
    torch.save(
        dict(initial=initial, args=args, expected=expected), tmp_path / "oracle.pt"
    )
    program = build_step(
        model,
        objective=objective,
        optimizer=AdamW(
            model.parameters(),
            lr=torch.tensor(0.001),
            opt_state_dtype=dtype,
            gradient_dtype=dtype,
            parameter_rounding="stochastic" if stochastic else "nearest",
            opt_state_rounding="stochastic" if stochastic else "nearest",
        ),
        example_args=args,
        partition="whole",
    )
    program.save(tmp_path / "program.ssp")
    program = Program.load(tmp_path / "program.ssp")
    assert program.bindings["optimizer_zero_lr_preserves_state"]
    profile = measure_program(program, warmup_seconds=0.001, measure_seconds=0.002)
    plan_program(
        apply_profiles(program, profile),
        memory_budgets=MemoryBudgets(2 << 30, 64 << 20),
        communication_info=CommunicationInfo(1e10, 1e10),
    ).save(tmp_path / "plan.ssp")
    child = subprocess.run(
        [
            sys.executable,
            "-S",
            str(Path(__file__).with_name("_diagnostics.py")),
            str(tmp_path),
        ],
        env={
            **os.environ,
            "TORCHINDUCTOR_CACHE_DIR": str(tmp_path / "worker-inductor"),
            "TRITON_CACHE_DIR": str(tmp_path / "worker-triton"),
        },
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    assert "Qualified saved startup diagnostics" in child.stdout
