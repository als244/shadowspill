"""Lifetime selectors stop at build; final composed tasks execute cold."""

import pytest
import torch

from engine_tests.build.test_training import objective
from engine_tests.execution.test_chunked_saved_plan import _verify_saved
from engine_tests.execution.test_region_saved_plan import MixerBlock, RegionNetwork
from engine_tests.pytorch.chunking.test_regions import RegionChunks
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


class DelayedBlock(MixerBlock):
    def forward(self, x):
        early = x.sin()
        q = x @ self.weight + (x @ self.a) @ self.b
        return torch.nn.functional.silu(q.softmax(1) @ self.out) + early


class DelayedNetwork(RegionNetwork):
    def __init__(self, lora):
        torch.nn.Module.__init__(self)
        self.blocks = torch.nn.ModuleList([DelayedBlock(lora), DelayedBlock(lora)])


class PackedBlock(MixerBlock):
    def __init__(self, lora):
        super().__init__(lora)
        self.out = torch.nn.Parameter(torch.randn(8, 8) / 8)

    def forward(self, x):
        early = x.sin()
        packed = x @ self.weight + (x @ self.a) @ self.b
        q, gate = packed.split(8, dim=-1)
        return torch.nn.functional.silu(q.softmax(1) @ self.out) * gate.sin() + early


class PackedNetwork(RegionNetwork):
    def __init__(self, lora):
        torch.nn.Module.__init__(self)
        self.blocks = torch.nn.ModuleList([PackedBlock(lora), PackedBlock(lora)])


class NormalizedPackedBlock(PackedBlock):
    def forward(self, x):
        normalized = x.sin()
        packed = normalized @ self.weight + (normalized @ self.a) @ self.b
        q, gate = packed.split(8, dim=-1)
        return (
            torch.nn.functional.silu(q.softmax(1) @ self.out) * gate.sin() + normalized
        )


class NormalizedPackedNetwork(RegionNetwork):
    def __init__(self, lora):
        torch.nn.Module.__init__(self)
        self.blocks = torch.nn.ModuleList(
            [
                NormalizedPackedBlock(lora),
                NormalizedPackedBlock(lora),
            ]
        )


@pytest.mark.cuda
@pytest.mark.parametrize(
    "lora,retain_sequence,compact_views",
    # Both forms and both view settings run in the default suite; the other
    # two combinations run with `-m slow`.
    [
        pytest.param(False, False, False, marks=pytest.mark.slow),
        (False, False, True),
        (True, True, False),
        pytest.param(True, True, True, marks=pytest.mark.slow),
    ],
)
def test_public_lifetimes_profile_plan_and_train_without_serializing_selector(
    tmp_path, lora, retain_sequence, compact_views, replay_work=False
):
    if not torch.cuda.is_available():
        pytest.skip("requires an accelerator for task finalization")
    torch.manual_seed(731)
    network = PackedNetwork if compact_views else DelayedNetwork
    if replay_work:
        network = NormalizedPackedNetwork
    model = network(lora)
    with torch.device("meta"):
        template = network(lora)
    examples = [
        ((torch.randn(2, 17, 8),), {"labels": torch.randn(2, 17, 8)}) for _ in range(2)
    ]
    building = True
    calls = []

    def select(operation):
        assert building, "selector ran after build"
        calls.append((operation.stage_id, operation.index))
        if operation.target == "aten.sin.default":
            return (
                LifetimeSelection(replay=True)
                if replay_work
                else LifetimeSelection(move=True)
            )
        if compact_views and operation.target in (
            "aten.view.default",
            "aten._unsafe_view.default",
            "aten.slice.Tensor",
            "aten.split.Tensor",
            "<built-in function getitem>",
        ):
            return LifetimeSelection(compact=True)
        return None

    barriers = ("aten._softmax.default", "aten._softmax_backward_data.default")
    if compact_views:
        # The softmax VJP uses its output, so both packed inputs have already
        # died by that barrier. Protect the intervening activation as well:
        # only the gate slice still needs the packed projection there.
        barriers += ("aten.silu.default",)
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
    if not replay_work:
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
    if compact_views:
        assert any(
            decision["applied"]
            for report in reports
            for decision in report.get("compaction", {}).get("decisions", ())
        )
        assert all(
            after <= before
            for report in reports
            for _, before, after in report.get("compaction", {}).get(
                "barrier_bytes", ()
            )
        )
    if replay_work:
        assert any(
            decision["applied"]
            for report in reports
            for decision in report.get("replay", {}).get("decisions", ())
        )
        assert any(report.get("replay", {}).get("copies") for report in reports)
        assert all(
            after <= before
            for report in reports
            for _, before, after in report.get("replay", {}).get("barrier_bytes", ())
        )
    before = tuple(calls)
    _verify_saved(program, model, examples, "step", tmp_path)
    assert tuple(calls) == before


@pytest.mark.cuda
@pytest.mark.parametrize("lora,retain_sequence", [(False, False), (True, True)])
def test_public_replay_and_compaction_profile_plan_and_train_cold(
    tmp_path, lora, retain_sequence
):
    test_public_lifetimes_profile_plan_and_train_without_serializing_selector(
        tmp_path, lora, retain_sequence, compact_views=True, replay_work=True
    )
