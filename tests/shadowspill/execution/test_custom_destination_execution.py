"""Public builds preserve custom gradient destinations through saved execution."""

import copy

import pytest
import torch

from shadowspill import build_step
from shadowspill.pytorch import (
    AxisChunks,
    ChunkRegions,
    IsolateOperations,
    OperatorRule,
)
from shadowspill.task.inputs import TaskInputRole
from tests.shadowspill.build.test_training import objective
from tests.shadowspill.execution.test_chunked_saved_plan import _verify_saved
from tests.shadowspill.pytorch.capture._gradient_destination_ops import (
    interleave,
    projection,
)
from tests.shadowspill.pytorch.capture.test_gradient_destinations import RULE
from tests.shadowspill.pytorch.chunking.test_custom_destinations import AXIS_RULES

PROJECTION = "gradient_destination_test.projection.default"
RULES = (*AXIS_RULES, OperatorRule(PROJECTION, "1", (0,)))


class Inner:
    def configure_stage(self, stage):
        axes = {
            i: domains["positions"]
            for i, domains in enumerate(stage.input_domains)
            if "positions" in domains
        }
        return AxisChunks(4, axes, domain="positions", rules=RULES) if axes else None


class Chunks:
    def __init__(self, variant):
        self.variant = variant

    def configure_stage(self, stage):
        if stage.terminal:
            return None
        axes = {
            i: 0
            for i, (value, role) in enumerate(
                zip(stage.inputs, stage.input_roles, strict=True)
            )
            if value is not None
            and len(value.shape) == 2
            and role not in {TaskInputRole.PARAMETER, TaskInputRole.BUFFER}
        }
        return (
            AxisChunks(
                9,
                axes,
                variants=(self.variant,),
                domain="positions",
                rules=RULES,
                regions=ChunkRegions(
                    IsolateOperations(
                        (PROJECTION, RULE.target, RULE.replacement), base="whole"
                    ),
                    Inner(),
                    variants=(self.variant,),
                ),
            )
            if axes
            else None
        )


class Network(torch.nn.Module):
    def __init__(self, opaque):
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(3, 7) * 0.1)
        self.b = torch.nn.Parameter(torch.randn(10, 3) * 0.1)
        self.opaque = opaque

    def forward(self, x):
        def project(value):
            if self.opaque:
                return projection(value, self.a, self.b)
            return ((value @ self.a.T) @ interleave(self.b).T) * 1.75

        return project(x) + project(x + 0.25)


@pytest.mark.cuda
@pytest.mark.parametrize("variant", ["save", "recompute"])
def test_shared_opaque_gradients_in_cold_saved_step(tmp_path, variant):
    torch.manual_seed(7301)
    model = Network(opaque=False)
    template = copy.deepcopy(model).to("meta")
    template.opaque = True
    examples = [
        ((torch.randn(rows, 7),), {"labels": torch.randn(rows, 10)})
        for rows in (23, 18)
    ]
    program = build_step(
        template,
        optimizer=torch.optim.AdamW(template.parameters(), foreach=False),
        objective=objective,
        example_calls=examples,
        partition=IsolateOperations((PROJECTION,), base="whole"),
        chunking=Chunks(variant),
        gradient_destination_rules=(RULE,),
    )
    assert any(
        dependency["operator"] == "gradient_destination_test::accumulating"
        for dependencies in program.bindings["operator_dependencies"].values()
        for dependency in dependencies
    )
    # The oracle uses ordinary autograd for all nonzero factors. The worker
    # reloads the saved Plan with empty compiler caches and preparation blocked.
    _verify_saved(program, model, examples, "step", tmp_path)
