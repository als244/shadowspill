"""Saved stochastic computation consumes only execution-time generator state."""

import sys
from pathlib import Path

import torch

from shadowspill import Executor, Plan
from tests.shadowspill.execution._cold import forbid_preparation

forbid_preparation("tests.shadowspill.execution.test_random")
directory = Path(sys.argv[1])
oracle = torch.load(directory / "oracle.pt", weights_only=True)


def fill(request, destination):
    value = oracle["initial"][request.name].reshape(-1)
    destination.copy_(value[request.offset : request.offset + destination.numel()])


with Executor(Plan.load(directory / "plan.ssp")) as executor:
    executor.initialize(fill, chunk_bytes=40)
    torch.manual_seed(916)
    cpu, device = torch.get_rng_state(), torch.cuda.get_rng_state()

    def run():
        return executor.run(args=[(oracle["input"],)]).result().cpu()

    first = run()
    assert not torch.equal(torch.cuda.get_rng_state(), device)
    assert torch.equal(torch.get_rng_state(), cpu)
    torch.cuda.set_rng_state(device)
    second = run()
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert not torch.equal(second, run())
print("Qualified three saved-plan invocations", flush=True)
