"""Nested execution follows the saved build/profile/plan/execute contract."""

import pytest
import torch

from engine_tests.build.test_training import objective
from engine_tests.execution.test_chunked_saved_plan import (
    ChunkRows,
    HostControlNetwork,
    _verify_saved,
)
from engine_tests.pytorch.chunking.test_regions import RegionChunks
from shadowspill.build import build_forward, build_forward_backward, build_step
from shadowspill.pytorch import GraphPairPolicy, RetentionPolicy, RetentionRule
from shadowspill.pytorch.chunking import ChunkRegions
from shadowspill.pytorch.partition import IsolateOperations


class MixerBlock(torch.nn.Module):
    def __init__(self, lora):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(8, 16) / 8, requires_grad=not lora)
        self.a = torch.nn.Parameter(torch.randn(8, 4) / 8)
        self.b = torch.nn.Parameter(torch.randn(4, 16) / 8)
        self.out = torch.nn.Parameter(torch.randn(16, 8) / 8)

    def forward(self, x):
        q = (x @ self.weight + (x @ self.a) @ self.b).sin()
        return torch.nn.functional.silu(q.softmax(1) @ self.out) + x


class RegionNetwork(torch.nn.Module):
    def __init__(self, lora):
        super().__init__()
        self.blocks = torch.nn.ModuleList([MixerBlock(lora), MixerBlock(lora)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


@pytest.mark.cuda
@pytest.mark.parametrize(
    "variant, retain_sequence, lora, optimizer",
    [
        pytest.param("recompute", False, False, False, id="full-backward-recompute"),
        pytest.param("save", False, False, True, id="full-step-save"),
        pytest.param("recompute", True, False, True, id="full-step-retained"),
        pytest.param("recompute", False, True, True, id="lora-step-recompute"),
        pytest.param("save", False, True, False, id="lora-backward-save"),
        pytest.param("recompute", True, True, False, id="lora-backward-retained"),
    ],
)
def test_regions_retain_outer_checkpoints_and_train_after_cold_reload(
    tmp_path, variant, retain_sequence, lora, optimizer
):
    # Cover every pair of retention choice, full/LoRA and backward/step without
    # rebuilding all twelve products. CPU region VJPs cover the full/LoRA x
    # save/recompute x geometry product; maintained DP cases additionally cover
    # retained LoRA with optimizer state. All saved execution assertions remain.
    if not torch.cuda.is_available():
        pytest.skip("requires an accelerator for task finalization")
    torch.manual_seed(731)
    model = RegionNetwork(lora)
    with torch.device("meta"):
        template = RegionNetwork(lora)
    examples = [((torch.randn(2, 17, 8),), {"labels": torch.randn(2, 17, 8)})] * 2
    policy = ChunkRegions(
        IsolateOperations(
            ("aten._softmax.default", "aten._softmax_backward_data.default"),
            base="whole",
        ),
        RegionChunks(17, 2),
        variants=(variant,),
    )
    common = dict(
        objective=objective,
        example_calls=examples,
        partition="auto",
        chunking=policy,
        graph_pair_policy=GraphPairPolicy(
            recompute=RetentionPolicy(
                default="recompute",
                rules=(RetentionRule("aten._softmax.default", "retain"),)
                if retain_sequence
                else (),
            )
        ),
    )
    program = (
        build_step(
            template,
            optimizer=torch.optim.AdamW(template.parameters(), foreach=False),
            **common,
        )
        if optimizer
        else build_forward_backward(template, **common)
    )
    metadata = program.bindings["task_metadata"]
    regions = [
        item["execution_regions"]
        for item in metadata.values()
        if "execution_regions" in item
    ]
    assert regions
    # Retaining the sequence result removes its forward replay from backward.
    assert all(len(item["regions"]) >= 3 for item in regions)
    assert {
        child["row_chunks"]["chunk_size"]
        for item in regions
        for child in item["regions"]
        if "row_chunks" in child
    } == {3, 7}
    _verify_saved(
        program, model, examples, "step" if optimizer else "forward_backward", tmp_path
    )


@pytest.mark.cuda
def test_region_forward_preserves_host_controls_after_cold_reload(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("requires an accelerator for task finalization")
    torch.manual_seed(731)
    model = HostControlNetwork()
    with torch.device("meta"):
        template = HostControlNetwork()
    examples = [((torch.randn(2, 17, 8),), {})]
    program = build_forward(
        template,
        example_args=examples[0][0],
        chunking=ChunkRegions(
            IsolateOperations(("aten.sin.default",), base="whole"),
            ChunkRows("recompute"),
        ),
    )
    _verify_saved(
        program, model, examples, "forward", tmp_path, host_output_leaves=(1,)
    )
