"""Packed-axis singleton tails survive the complete saved training pipeline."""

import pytest
import torch

from shadowspill import build_forward_backward
from shadowspill.pytorch import AxisChunks, ChunkRegions, IsolateOperations
from shadowspill.task.inputs import TaskInputRole
from tests.shadowspill.build.test_training import objective
from tests.shadowspill.execution.test_chunked_saved_plan import _verify_saved
from tests.shadowspill.pytorch.chunking.test_composition import Inner


class PackedProjection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(15, 8) / 8)

    def forward(self, x):
        packed = x.permute(2, 0, 1, 3).reshape(3, -1, 5)
        tokens = packed.permute(1, 0, 2).reshape(1, -1, 15)
        return (tokens @ self.weight).sin()


class ProjectionChunks:
    def __init__(self, variant):
        self.variant = variant

    def configure_stage(self, stage):
        if stage.terminal:
            return None
        axes = {
            index: 1
            for index, (value, role) in enumerate(
                zip(stage.inputs, stage.input_roles, strict=True)
            )
            if value is not None
            and len(value.shape) in (3, 4)
            and role not in {TaskInputRole.PARAMETER, TaskInputRole.BUFFER}
        }
        return (
            AxisChunks(
                8,
                axes,
                variants=(self.variant,),
                domain="positions",
                regions=ChunkRegions("whole", Inner(), variants=(self.variant,)),
            )
            if axes
            else None
        )


@pytest.mark.cuda
@pytest.mark.parametrize("variant", ["save", "recompute"])
def test_singleton_tails_reload_and_accumulate_gradients(tmp_path, variant):
    if not torch.cuda.is_available():
        pytest.skip("requires an accelerator for task finalization")
    torch.manual_seed(731)
    model = PackedProjection()
    examples = [
        ((torch.randn(1, extent, 3, 5),), {"labels": torch.randn(1, extent, 8)})
        for extent in (17, 9)
    ]
    with torch.device("meta"):
        template = PackedProjection()
    program = build_forward_backward(
        template,
        objective=objective,
        example_calls=examples,
        partition=IsolateOperations(("aten.sin.default",), base="whole"),
        chunking=ProjectionChunks(variant),
    )
    assert any(
        metadata.get("row_chunks", {}).get("chunk_size") == 8
        for metadata in program.bindings["task_metadata"].values()
    )
    _verify_saved(program, model, examples, "forward_backward", tmp_path)
