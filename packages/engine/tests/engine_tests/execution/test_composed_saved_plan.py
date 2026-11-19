"""Nested axis/region compositions survive the complete saved training pipeline."""

import pytest
import torch

from engine_tests.build.test_training import objective
from engine_tests.execution.test_chunked_saved_plan import _verify_saved
from engine_tests.execution.test_region_saved_plan import MixerBlock
from engine_tests.pytorch.chunking.test_composition import Inner
from shadowspill.build import build_step
from shadowspill.pytorch.chunking import AxisChunks, ChunkRegions
from shadowspill.pytorch.partition import IsolateOperations


class ComposedBlock(MixerBlock):
    def forward(self, x):
        q = x @ self.weight + (x @ self.a) @ self.b
        return torch.nn.functional.silu(q) @ self.out + x


class ComposedNetwork(torch.nn.Module):
    def __init__(self, lora):
        super().__init__()
        self.blocks = torch.nn.ModuleList([ComposedBlock(lora), ComposedBlock(lora)])
        self.blocks[1].out = self.blocks[0].out

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ComposedChunks:
    def __init__(self, variant):
        self.variant = variant

    def configure_stage(self, stage):
        if stage.terminal:
            return None
        axes = {
            i: 1
            for i, value in enumerate(stage.inputs)
            if value is not None and len(value.shape) == 3
        }
        return (
            AxisChunks(
                11,
                axes,
                variants=(self.variant,),
                domain="positions",
                regions=ChunkRegions(
                    IsolateOperations(
                        ("aten.silu.default", "aten.silu_backward.default"),
                        base="whole",
                    ),
                    Inner(),
                ),
            )
            if axes
            else None
        )


@pytest.mark.cuda
@pytest.mark.parametrize(
    "lora,variant",
    # Both LoRA settings and both variants run in the default suite; the
    # other two combinations run with `-m slow`.
    [
        (False, "recompute"),
        pytest.param(False, "save", marks=pytest.mark.slow),
        pytest.param(True, "recompute", marks=pytest.mark.slow),
        (True, "save"),
    ],
)
def test_nested_chunks_train_with_tied_weights_after_cold_reload(
    tmp_path, variant, lora
):
    if not torch.cuda.is_available():
        pytest.skip("requires an accelerator for task finalization")
    torch.manual_seed(731)
    model = ComposedNetwork(lora)
    with torch.device("meta"):
        template = ComposedNetwork(lora)
    examples = [
        ((torch.randn(2, extent, 8),), {"labels": torch.randn(2, extent, 8)})
        for extent in (29, 22)
    ]
    program = build_step(
        template,
        optimizer=torch.optim.AdamW(template.parameters(), foreach=False),
        objective=objective,
        example_calls=examples,
        chunking=ComposedChunks(variant),
    )
    compositions = [
        item["row_chunks"]
        for item in program.bindings["task_metadata"].values()
        if "row_chunks" in item
    ]
    assert compositions and all(item["child_regions"] for item in compositions)
    assert any(item["child_regions"]["accumulators"] for item in compositions)
    _verify_saved(program, model, examples, "step", tmp_path)
