"""Caller outputs may view inputs or parameters; subsequent calls must preserve them."""

import pytest
import torch
from torch import nn
from torch.utils._pytree import tree_map

from shadowspill import build_forward
from tests.shadowspill.execution.helpers import run_worker, save_plan


class Aliases(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(8, 8))

    def forward(self, x):
        return {
            "input": x,
            "view": x[:, ::2],
            "weight": self.weight,
            "computed": x @ self.weight,
            "constant": torch.tensor([1.25, 2.5, 0.75], device=x.device),
        }


@pytest.mark.cuda
def test_alias_outputs_survive_repeated_calls(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    model = Aliases()
    x = torch.randn(4, 8)
    with torch.device("meta"):
        template = Aliases()
    program = build_forward(template, example_args=(x,), partition="whole")
    save_plan(program, tmp_path)
    outputs = tree_map(lambda t: t.detach().clone(), model(x))
    state = {"weight": model.weight.detach().clone()}
    for source in program.bindings["root_sources"]:
        if source["kind"] == "CONSTANT_TENSOR":
            state["constant." + source["name"]] = torch.tensor(source["value"])
    torch.save(
        dict(
            initial=model.state_dict(),
            examples=[((x,), {})],
            expected=[(outputs, state)] * 3,
            hyperparams=[{}] * 3,
            restore=False,
            accumulate=[False] * 3,
        ),
        tmp_path / "oracle.pt",
    )
    run_worker(tmp_path)
