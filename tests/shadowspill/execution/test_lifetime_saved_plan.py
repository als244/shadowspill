"""Lifetime selectors stop at build; final composed tasks execute cold."""

import pytest
import torch

from shadowspill.build import build_step
from shadowspill.pytorch import (
    ChunkRegions,
    ExecutionLifetimes,
    GraphPairPolicy,
    LifetimeRule,
    LifetimeSelection,
    RetentionPolicy,
    RetentionRule,
)
from shadowspill.pytorch.partition import IsolateOperations
from tests.shadowspill.build.test_training import objective
from tests.shadowspill.execution.test_chunked_saved_plan import _verify_saved
from tests.shadowspill.execution.test_region_saved_plan import MixerBlock, RegionNetwork
from tests.shadowspill.pytorch.chunking.test_regions import RegionChunks


class DelayedBlock(MixerBlock):
    def forward(self, x):
        early = x.sin()
        q = x @ self.weight + (x @ self.a) @ self.b
        return torch.nn.functional.silu(q.softmax(1) @ self.out) + early


class DelayedNetwork(RegionNetwork):
    def __init__(self, lora):
        torch.nn.Module.__init__(self)
        self.blocks = torch.nn.ModuleList([DelayedBlock(lora), DelayedBlock(lora)])


@pytest.mark.cuda
@pytest.mark.parametrize("lora,retain_sequence", [(False, False), (True, True)])
def test_public_lifetimes_profile_plan_and_train_without_serializing_selector(
    tmp_path, lora, retain_sequence
):
    if not torch.cuda.is_available():
        pytest.skip("requires an accelerator for task finalization")
    torch.manual_seed(731)
    model = DelayedNetwork(lora)
    with torch.device("meta"):
        template = DelayedNetwork(lora)
    examples = [
        ((torch.randn(2, 17, 8),), {"labels": torch.randn(2, 17, 8)}) for _ in range(2)
    ]
    building = True
    calls = []

    def select(operation):
        assert building, "selector ran after build"
        calls.append((operation.stage_id, operation.index))
        if operation.target == "aten.sin.default":
            return LifetimeSelection(move=True)
        return None

    barriers = ("aten._softmax.default", "aten._softmax_backward_data.default")
    program = build_step(
        template,
        optimizer=torch.optim.AdamW(template.parameters(), foreach=False),
        objective=objective,
        example_calls=examples,
        partition="auto",
        chunking=ChunkRegions(
            IsolateOperations(barriers, base="whole"),
            RegionChunks(17, 2),
            variants=("recompute",),
            lifetimes=ExecutionLifetimes(
                rules=tuple(LifetimeRule(target, barrier=True) for target in barriers),
                select=select,
            ),
        ),
        graph_pair_policy=GraphPairPolicy(
            recompute=RetentionPolicy(
                default="recompute",
                rules=(RetentionRule(barriers[0], "retain"),)
                if retain_sequence
                else (),
            )
        ),
    )
    building = False
    assert calls
    reports = [
        report
        for metadata in program.bindings["task_metadata"].values()
        for report in metadata.get("execution_lifetimes", ())
    ]
    assert reports
    assert any(
        decision["applied"]
        for report in reports
        for decision in report["schedule"]["decisions"]
    )
    assert all(
        after <= before
        for report in reports
        for _, before, after in report["schedule"]["barrier_bytes"]
    )
    before = tuple(calls)
    _verify_saved(program, model, examples, "step", tmp_path)
    assert tuple(calls) == before
