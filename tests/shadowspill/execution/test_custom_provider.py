"""A saved task restores its operator registrations in a fresh model-free worker."""

import pytest
import torch
from torch import nn

from shadowspill import build_forward, build_step
from shadowspill.pytorch.dependencies import operator_dependencies, restore_dependencies
from tests.shadowspill.build.test_training import objective
from tests.shadowspill.execution._custom_provider import square
from tests.shadowspill.execution._oracle import oracle
from tests.shadowspill.execution.helpers import run_worker, save_plan


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
@pytest.mark.parametrize("training", [False, True], ids=["forward", "step"])
def test_custom_provider_cold_registration(tmp_path, training):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    model = Model()
    x = torch.randn(4, 8)
    with torch.device("meta"):
        template = Model()
    examples = [((x,), {"labels": torch.randn_like(x)}) for _ in range(2)]
    if training:
        program = build_step(
            template,
            optimizer=torch.optim.AdamW(template.parameters(), lr=0.001, foreach=False),
            objective=objective,
            example_args=examples[0][0],
            example_kwargs=examples[0][1],
            num_examples=2,
            partition="whole",
        )
    else:
        examples = [((x,), {})]
        program = build_forward(template, example_args=(x,), partition="whole")
    dependencies = [
        item
        for items in program.bindings["operator_dependencies"].values()
        for item in items
    ]
    assert {item["module"] for item in dependencies} == {
        "tests.shadowspill.execution._custom_provider"
    }
    with pytest.raises(ValueError, match="registration source differs"):
        restore_dependencies([{**dependencies[0], "module_sha256": "bad"}])
    save_plan(program, tmp_path)
    torch.save(
        oracle(model, examples, "step" if training else "forward"),
        tmp_path / "oracle.pt",
    )
    run_worker(tmp_path)
