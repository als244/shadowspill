"""A generic provider buffer is recreated by each phase and cold Executor."""

import copy
import json

import pytest
import torch
from torch import nn

from shadowspill import (
    Resource,
    build_forward,
    build_forward_backward,
    build_step,
    measure_program,
)
from tests.shadowspill.build.test_training import objective
from tests.shadowspill.execution._oracle import oracle
from tests.shadowspill.execution.helpers import run_worker, save_plan
from tests.shadowspill.resources._provider import _active, scale


class Network(nn.Module):
    def __init__(self, *, reference=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(8))
        self.reference = reference

    def forward(self, value):
        value = self.weight * value
        if self.reference:
            return value * 4
        return scale(scale(value, "lookup"), "lookup")


@pytest.mark.cuda
@pytest.mark.parametrize("kind", ["forward", "forward_backward", "step"])
def test_saved_provider_resource_lifecycle_and_memory(tmp_path, kind):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.manual_seed(179)
    example = (
        (torch.randn(3, 8),),
        {} if kind == "forward" else {"labels": torch.randn(3, 8)},
    )
    reference = Network(reference=True)
    model = copy.deepcopy(reference).to("meta")
    model.reference = False
    options = {"events": str(tmp_path / "events"), "elements": 256 * 1024}
    common = dict(
        example_args=example[0],
        example_kwargs=example[1],
        partition="whole",
        resources={
            "lookup": Resource(
                "tests.shadowspill.resources._provider:initialize", options
            )
        },
    )
    if kind == "forward":
        program = build_forward(model, **common)
    elif kind == "forward_backward":
        program = build_forward_backward(model, objective=objective, **common)
    else:
        program = build_step(
            model,
            objective=objective,
            optimizer=torch.optim.AdamW(model.parameters(), foreach=False, lr=0.001),
            **common,
        )
    assert not _active
    profiles = measure_program(program, warmup_seconds=0.001, measure_seconds=0.002)
    assert not _active
    assert profiles.metadata["provider_resource_bytes"] >= options["elements"] * 4
    assert profiles.fixed_device_bytes >= profiles.metadata["provider_resource_bytes"]
    save_plan(program, tmp_path, profiles=profiles)
    torch.save(oracle(reference, [example], kind), tmp_path / "oracle.pt")
    run_worker(tmp_path)
    events = [
        json.loads(line) for line in (tmp_path / "events").read_text().splitlines()
    ]
    assert [event["event"] for event in events] == ["open", "close"] * 3
    assert len({event["pid"] for event in events}) == 2
