"""A saved task restores its operator registrations in a fresh model-free worker."""

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
    measure_program,
    plan_program,
)
from shadowspill.pytorch.dependencies import operator_dependencies, restore_dependencies
from tests.shadowspill.execution._custom_provider import square


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.full((8,), 0.5))

    def forward(self, x):
        return square(x * self.weight)


def indirect_call(op, value):
    return op(value)


def test_finds_operator_arguments_and_refuses_temporary_dispatch_changes():
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    node = graph.call_function(indirect_call, (square._opoverload, x))
    graph.output(node)
    module = torch.fx.GraphModule(nn.Module(), graph)
    assert operator_dependencies(module)[0]["operator"] == "shadowspill_fixture::square"
    with (
        square.set_kernel_enabled("cpu", enabled=False),
        pytest.raises(ValueError, match="temporarily disabled"),
    ):
        operator_dependencies(module)


@pytest.mark.cuda
def test_custom_provider_cold_registration(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    model = Model()
    x = torch.randn(4, 8)
    with torch.device("meta"):
        template = Model()
    program = build_forward(template, example_args=(x,), partition="whole")
    dependencies = [
        item
        for items in program.bindings["operator_dependencies"].values()
        for item in items
    ]
    assert len(dependencies) == 1
    assert dependencies[0]["module"] == "tests.shadowspill.execution._custom_provider"
    with pytest.raises(ValueError, match="registration source differs"):
        restore_dependencies([{**dependencies[0], "module_sha256": "bad"}])
    profiles = measure_program(program, warmup_seconds=0.001, measure_seconds=0.002)
    plan = plan_program(
        apply_profiles(program, profiles),
        memory_budgets=MemoryBudgets(2 << 30, 64 << 20),
        communication_info=CommunicationInfo(1e10, 1e10),
    )
    plan.save(tmp_path / "plan.ssp")
    expected = model(x).detach()
    state = {"weight": model.weight.detach().clone()}
    torch.save(
        dict(
            initial=state,
            examples=[((x,), {})],
            expected=[(expected, state)] * 3,
            hyperparams=[{}] * 3,
            accumulate=[False] * 3,
            restore=False,
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
