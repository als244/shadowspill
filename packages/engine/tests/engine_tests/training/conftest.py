"""CPU loop fixtures; saved-plan GPU execution is tested separately."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from engine_tests.execution.test_checkpoint import MemoryExecutor
from engine_tests.offline import _example
from shadowspill import MemoryBudgets, apply_profiles, plan_program
from shadowspill.execution import TrainingResult


@pytest.fixture
def executor():
    program = replace(
        _example.program(),
        bindings={
            "kind": "step",
            "optimizer_bindings": [{"name": "lr", "role": "hyperparameter"}],
        },
    )
    plan = plan_program(
        apply_profiles(program, _example.profiles()),
        memory_budgets=MemoryBudgets(100, 64),
        communication_info=_example.communication(),
        search_options=_example.options(),
    )
    result = MemoryExecutor()
    result.plan, result.device = plan, torch.device("cpu")
    result.calls, result.closed = [], False
    result.parameter_norms = True
    result.synchronize = lambda: None
    result.close = lambda: setattr(result, "closed", True)

    def run(**inputs):
        result.calls.append(inputs)
        output = TrainingResult(
            (torch.tensor(2.0),),
            ({"observed": torch.tensor(3.0)},),
            {
                "weight": {
                    "param_norm": torch.tensor(4.0),
                    "grad_norm": torch.tensor(5.0),
                }
            }
            if inputs.get("collect_parameter_norms")
            else {},
        )
        return SimpleNamespace(result=lambda: output)

    result.run = run
    return result
