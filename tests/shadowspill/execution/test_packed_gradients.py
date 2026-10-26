"""Packed projection gradients must remain independent optimizer inputs."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from shadowspill import build_step
from tests.shadowspill.execution.test_tensor_representations import objective, save_plan


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Parameter(torch.randn(4, 8))
        self.second = nn.Parameter(torch.randn(4, 8))

    def forward(self, value):
        return nn.functional.linear(value, torch.cat((self.first, self.second)))


@pytest.mark.cuda
def test_packed_parameter_gradients_in_cold_worker(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.manual_seed(127)
    reference, sample = Network(), torch.randn(3, 8)
    initial = {n: t.detach().clone() for n, t in reference.named_parameters()}
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.01, foreach=False)
    expected = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = objective(reference, sample)
        loss.backward()
        optimizer.step()
        expected.append(
            (
                [(loss.detach(), None)],
                {n: t.detach().clone() for n, t in reference.named_parameters()},
            )
        )
    with torch.device("meta"):
        template = Network()
    program = build_step(
        template,
        optimizer=torch.optim.SGD(template.parameters(), lr=0.01, foreach=False),
        objective=objective,
        example_args=(sample,),
        partition="whole",
    )
    save_plan(program, tmp_path / "plan.ssp")
    torch.save(
        dict(
            initial=initial,
            examples=[((sample,), {})],
            expected=expected,
            hyperparams=[{}] * 3,
            accumulate=[False] * 3,
            restore=False,
        ),
        tmp_path / "oracle.pt",
    )
    worker = subprocess.run(
        [
            sys.executable,
            "-S",
            str(Path(__file__).with_name("_execute.py")),
            str(tmp_path),
        ],
        env={
            **os.environ,
            "TRITON_CACHE_DIR": str(tmp_path / "empty-triton"),
            "TORCHINDUCTOR_CACHE_DIR": str(tmp_path / "empty-inductor"),
        },
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert worker.returncode == 0, worker.stdout + worker.stderr
