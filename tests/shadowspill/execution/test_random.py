"""Preparation preserves caller RNG; real invocation consumes and replays it."""

import pytest
import torch
from torch import nn

from shadowspill import build_forward
from tests.shadowspill.execution.helpers import run_worker, save_plan


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(8, 8)

    def forward(self, x):
        return self.layer(x) + torch.rand_like(x)


@pytest.mark.cuda
def test_build_and_profile_preserve_rng_and_cold_execution_replays_it(tmp_path):
    torch.manual_seed(1009)
    model, x = Network(), torch.ones(3, 8)
    torch.cuda.init()
    cpu, device = torch.get_rng_state(), torch.cuda.get_rng_state()
    program = build_forward(model, example_args=(x,))
    assert torch.equal(torch.get_rng_state(), cpu)
    assert torch.equal(torch.cuda.get_rng_state(), device)
    save_plan(program, tmp_path)
    assert torch.equal(torch.get_rng_state(), cpu)
    assert torch.equal(torch.cuda.get_rng_state(), device)
    torch.save({"initial": model.state_dict(), "input": x}, tmp_path / "oracle.pt")
    run_worker(tmp_path, "_rng_execute.py")
