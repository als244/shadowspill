"""Saving CUDA literal values on CPU must not change compiled input placement."""

import pytest
import torch
from torch import nn

from shadowspill import Program, build_forward, build_forward_backward, measure_program


class LiteralModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(4, 4))

    def forward(self, x):
        coefficients = torch.tensor([2, 3, 5, 7], device=x.device, dtype=torch.int32)
        return (x @ self.weight) * coefficients


def objective(model, x):
    return model(x).sum()


@pytest.mark.cuda
@pytest.mark.parametrize("training", [False, True])
def test_literal_geometry_and_host_values_survive_example_rebinding(tmp_path, training):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    with torch.device("meta"):
        model = LiteralModel()
    options = dict(example_args=(torch.zeros(4, 4),), partition="whole")
    if training:
        program = build_forward_backward(
            model, objective=objective, num_examples=2, **options
        )
    else:
        program = build_forward(model, **options)
    program.save(tmp_path / "program.ssp")
    program = Program.load(tmp_path / "program.ssp")
    constants = [
        s for s in program.bindings["root_sources"] if s["kind"] == "CONSTANT_TENSOR"
    ]
    assert len(constants) == 1 and tuple(constants[0]["value"]) == (2, 3, 5, 7)
    found = 0
    for description in program.bindings["task_inputs"].values():
        geometry = dict(
            zip(
                description["tensor_positions"],
                description["tensor_geometry"],
                strict=True,
            )
        )
        for position, source in enumerate(description["provenance"]):
            if source["role"] == "constant":
                assert geometry[position]["device_type"] == "cuda"
                assert tuple(source["value"]) == (2, 3, 5, 7)
                found += 1
    assert found
    profiles = measure_program(program, warmup_seconds=0.001, measure_seconds=0.002)
    assert profiles.tasks
