"""Repeated saved forward tasks preserve owned outputs and shared mutation order."""

from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.utils._pytree import tree_map

from engine_tests.execution.helpers import run_worker
from shadowspill import (
    CommunicationInfo,
    MemoryBudgets,
    apply_profiles,
    build_forward,
    instantiate_examples,
    measure_program,
    plan_program,
)


class CountedOutputs(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(4, 4))
        self.tied = self.weight
        self.register_buffer("counter", torch.zeros(1))

    def forward(self, x, scale):
        self.counter.add_(1)
        constant = torch.tensor([1.25, 0.5, -0.25, 0.75], device=x.device)
        value = (x @ self.tied + constant) * scale + self.counter
        return {
            "input": x,
            "view": x[:, ::2],
            "weight": self.weight,
            "counter": self.counter,
            "value": value,
            "nested": (value.sum(), "kept", None),
            "constant": constant,
        }


@pytest.mark.cuda
def test_saved_forward_counts_in_cold_worker(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    calls = [
        tuple(
            (torch.arange(8).reshape(2, 4).float() / 16 + offset, torch.tensor(scale))
            for offset, scale in values
        )
        for values in (
            ((0.0, 1.0), (0.25, -0.5), (0.5, 2.0)),
            ((-0.25, 0.0),),
            ((0.75, 0.125), (1.0, -1.0)),
        )
    ]
    with torch.device("meta"):
        template = CountedOutputs()
    source = build_forward(template, example_args=calls[0][0], partition="whole")
    source = replace(source, bindings={**source.bindings, "operation_mode": "training"})
    profile = measure_program(source, warmup_seconds=0.001, measure_seconds=0.002)
    count = instantiate_examples(source, [0, 0, 0])
    repeated = instantiate_examples(count, [2, 0])
    for name, program in (("count", count), ("single", source), ("repeated", repeated)):
        assert program.executables == source.executables
        plan = plan_program(
            apply_profiles(program, profile),
            memory_budgets=MemoryBudgets(2 << 30, 64 << 20),
            communication_info=CommunicationInfo(1e10, 1e10),
        )
        plan.save(tmp_path / f"{name}.ssp")
    source.save(tmp_path / "source.ssp")
    profile.save(tmp_path / "profiles.ssp")
    model = CountedOutputs()
    with torch.no_grad():
        model.weight.copy_(torch.arange(16).reshape(4, 4) / 32)
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    expected = []
    for index, examples in enumerate(calls):
        outputs = tuple(
            tree_map(
                lambda value: (
                    value.detach().clone() if isinstance(value, torch.Tensor) else value
                ),
                model(*args),
            )
            for args in examples
        )
        expected.append(outputs[0] if index == 1 else outputs)
    torch.save(
        dict(initial=initial, calls=calls, expected=expected), tmp_path / "oracle.pt"
    )
    run_worker(tmp_path, "_forward_instances.py")
