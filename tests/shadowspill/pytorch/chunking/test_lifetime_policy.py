"""Public lifetime choices preserve values, precision, identity and boundaries."""

import json
from dataclasses import FrozenInstanceError

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from shadowspill.errors import CaptureError
from shadowspill.pytorch import (
    AxisChunks,
    ChunkRegions,
    ExecutionLifetimes,
    LayoutIndependentOperator,
    LifetimeRule,
    LifetimeSelection,
    ReadonlyOperator,
)
from shadowspill.pytorch.capture.aot import capture_graph_pair
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.capture.source import SourceOperation, set_source_operation
from shadowspill.pytorch.chunking.regions import region_artifact
from shadowspill.pytorch.partition import IsolateOperations
from tests.shadowspill.pytorch.chunking.test_lifetime_analysis import opaque_readonly
from tests.shadowspill.pytorch.chunking.test_regions import (
    RegionChunks,
    region_executable,
)

SOFTMAX = ("aten._softmax.default", "aten._softmax_backward_data.default")


class NoChunks:
    def configure_stage(self, stage):
        return None


def artifact(path="model.blocks.0"):
    x = torch.randn(17, 5)

    def function(x):
        early = x.sin()
        barrier = x.softmax(0)
        return early.cos() + barrier

    captured = GraphArtifact.capture(
        kind="inference", graph_module=make_fx(function)(x), example_inputs=(x,)
    )
    for index, node in enumerate(captured.graph_module.graph.nodes):
        if node.op == "call_function":
            node.meta["nn_module_stack"] = {"block": (path, "Block")}
            set_source_operation(
                node, SourceOperation(index, "call_function", "fixture.project")
            )
    return captured, x


def compose(captured, lifetimes, stage_id="block"):
    return region_artifact(
        captured,
        ChunkRegions(
            IsolateOperations(SOFTMAX, base="whole"), NoChunks(), lifetimes=lifetimes
        ),
        stage_id=stage_id,
        variant="recompute",
    )


def test_rules_move_work_with_backing_evidence_and_unchanged_outer_storage():
    captured, x = artifact()
    original = captured.graph_module.code
    policy = ExecutionLifetimes(
        rules=(
            LifetimeRule(
                "aten.sin.default",
                move=True,
                source_target="fixture.project",
                module_pattern="model.blocks.*",
            ),
            LifetimeRule(SOFTMAX[0], barrier=True),
        )
    )
    transformed = compose(captured, policy)
    assert captured.graph_module.code == original
    assert transformed.storage_contract == captured.storage_contract
    schedule = transformed.regions.identity["lifetimes"]["schedule"]
    assert any(item["applied"] for item in schedule["decisions"])
    assert schedule["barrier_bytes"][0][1:] == (17 * 5 * 4, 0)
    actual = region_executable(transformed)(x)
    torch.testing.assert_close(actual[0], captured.graph_module(x), rtol=0, atol=0)
    assert len(transformed.regions.children) == 2
    json.dumps(transformed.regions.lifetime_report)


def test_callback_is_immutable_build_only_and_occurrences_do_not_change_identity():
    observed = []

    def select(info):
        with pytest.raises(FrozenInstanceError):
            info.index = 123
        if info.inputs and info.inputs[0] is not None:
            with pytest.raises(FrozenInstanceError):
                info.inputs[0].shape = (1,)
        observed.append(info)
        return (
            LifetimeSelection(move=True) if info.target == "aten.sin.default" else None
        )

    policy = ExecutionLifetimes(select=select)
    one, x = artifact("model.blocks.0")
    two, _ = artifact("model.blocks.7")
    first, second = compose(one, policy, "first"), compose(two, policy, "second")
    assert first.compatibility_digest == second.compatibility_digest
    assert first.regions.lifetime_report != second.regions.lifetime_report
    assert "model.blocks" not in json.dumps(first.regions.identity)
    assert len(observed) == 2 * len(one.operator_targets)
    before = len(observed)
    region_executable(first)(x)
    assert len(observed) == before


@pytest.mark.parametrize("scope", ["source", "module"])
def test_unmatched_or_unattributed_source_never_applies_a_scoped_rule(scope):
    captured, _ = artifact()
    kwargs = (
        {"source_target": "different"}
        if scope == "source"
        else {"module_pattern": "different.*"}
    )
    transformed = compose(
        captured,
        ExecutionLifetimes(
            rules=(LifetimeRule("aten.sin.default", move=True, **kwargs),)
        ),
    )
    assert not transformed.regions.identity["lifetimes"]["selections"]


@pytest.mark.parametrize("callback", [False, True])
def test_conflicting_explicit_choices_are_refused(callback):
    captured, _ = artifact()
    policy = ExecutionLifetimes(
        rules=(LifetimeRule("aten.sin.default", move=True),)
        + (
            ()
            if callback
            else (
                LifetimeRule("aten.sin.default", move=False, module_pattern="model.*"),
            )
        ),
        select=(
            lambda info: (
                LifetimeSelection(move=False)
                if info.target == "aten.sin.default"
                else None
            )
        )
        if callback
        else None,
    )
    with pytest.raises(CaptureError, match="conflicting lifetime move"):
        compose(captured, policy)


def test_protected_barriers_override_movement_permission():
    captured, _ = artifact()
    transformed = compose(
        captured,
        ExecutionLifetimes(
            rules=(LifetimeRule("aten.sin.default", move=True, barrier=True),)
        ),
    )
    decisions = transformed.regions.identity["lifetimes"]["schedule"]["decisions"]
    assert len(decisions) == 1 and not decisions[0]["applied"]
    assert decisions[0]["reason"] == "protected memory barrier"


def test_readonly_versions_enter_identity_and_unknown_providers_refuse_movement():
    x = torch.randn(17, 5)
    captured = GraphArtifact.capture(
        kind="inference",
        graph_module=make_fx(lambda x: opaque_readonly(x) + x.cos())(x),
        example_inputs=(x,),
    )
    target = "lifetime_analysis_test.readonly.default"
    rules = (LifetimeRule(target, move=True),)
    with pytest.raises(CaptureError, match="cannot move"):
        compose(captured, ExecutionLifetimes(rules=rules))
    first, second = [
        compose(
            captured,
            ExecutionLifetimes(
                rules=rules, readonly=(ReadonlyOperator(target, version),)
            ),
        )
        for version in ("v1", "v2")
    ]
    assert first.compatibility_digest != second.compatibility_digest
    for transformed in (first, second):
        torch.testing.assert_close(
            region_executable(transformed)(x)[0],
            captured.graph_module(x),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: LifetimeSelection(move=1),
        lambda: LifetimeSelection(replay="cheap"),
        lambda: LifetimeRule("aten.sin.default"),
        lambda: ExecutionLifetimes(rules=[]),
        lambda: ExecutionLifetimes(select=True),
        lambda: ExecutionLifetimes(
            readonly=(ReadonlyOperator("op", "v1"), ReadonlyOperator("op", "v2"))
        ),
        lambda: ChunkRegions("whole", NoChunks(), lifetimes=True),
        lambda: AxisChunks(
            7,
            {0: 0},
            domain="positions",
            regions=ChunkRegions("whole", NoChunks(), lifetimes=ExecutionLifetimes()),
        ),
    ],
)
def test_invalid_and_nested_controls_fail_before_capture(factory):
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_selector_must_return_a_structured_selection():
    captured, _ = artifact()
    with pytest.raises(CaptureError, match="return LifetimeSelection or None"):
        compose(captured, ExecutionLifetimes(select=lambda info: True))


@pytest.mark.parametrize("save", [False, True])
@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.parametrize("batch", [1, 2])
def test_lifetime_policy_preserves_saved_cut_and_independent_parameter_vjp(
    save, lora, batch
):
    torch.manual_seed(731)
    inputs = (
        torch.randn(batch, 17, 5, dtype=torch.float64, requires_grad=True),
        torch.randn(5, 9, dtype=torch.float64, requires_grad=not lora),
        torch.randn(5, 3, dtype=torch.float64, requires_grad=True),
        torch.randn(3, 9, dtype=torch.float64, requires_grad=True),
        torch.randn(9, 5, dtype=torch.float64, requires_grad=True),
    )

    def layer(x, weight, a, b, out):
        delayed = x.sin()
        q = x @ weight + (x @ a) @ b
        return torch.nn.functional.silu(q.softmax(1) @ out) + delayed

    expected = layer(*inputs)
    tangent = torch.randn_like(expected)
    reference = torch.autograd.grad(
        expected, [v for v in inputs if v.requires_grad], tangent
    )
    pair = capture_graph_pair(
        make_fx(layer)(*inputs),
        inputs,
        original_output=layer(*inputs),
        memory_budget=float(save),
    )
    lifetimes = ExecutionLifetimes(
        rules=(
            LifetimeRule("aten.sin.default", move=True),
            LifetimeRule("aten.cos.default", move=True),
            *(LifetimeRule(target, barrier=True) for target in SOFTMAX),
        )
    )
    policy = ChunkRegions(
        IsolateOperations(SOFTMAX, base="whole"),
        RegionChunks(17, batch),
        lifetimes=lifetimes,
    )
    forward, backward = [
        region_artifact(
            item,
            policy,
            stage_id="block",
            variant="save" if save else "recompute",
        )
        for item in (pair.forward, pair.backward)
    ]
    assert forward.storage_contract == pair.forward.storage_contract
    assert backward.storage_contract == pair.backward.storage_contract
    with torch.no_grad():
        actual = region_executable(forward)(*inputs)
        saved = actual[-pair.saved_value_count :] if pair.saved_value_count else ()
        gradients = region_executable(backward)(*saved, tangent)
    torch.testing.assert_close(actual[0], expected, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(
        tuple(g for g in gradients if g is not None), reference, rtol=1e-10, atol=1e-10
    )


def test_public_compaction_retains_checkpoint_contract_and_build_only_decisions():
    torch.manual_seed(511)
    inputs = (torch.randn(2, 17, 5), torch.randn(5, 12))

    def layer(x, weight):
        packed = x @ weight
        a, gate = packed[..., :8], packed[..., 8:]
        attended = a.cumsum(1)
        return attended[..., :4] * gate.sin()

    captured = GraphArtifact.capture(
        kind="inference", graph_module=make_fx(layer)(*inputs), example_inputs=inputs
    )
    building = True
    calls = []

    def select(info):
        assert building
        calls.append(info.index)
        if info.target == "aten.slice.Tensor":
            return LifetimeSelection(compact=True)
        return None

    barriers = ("aten.cumsum.default", "aten.sin.default")
    policy = ChunkRegions(
        IsolateOperations(barriers, base="whole"),
        NoChunks(),
        lifetimes=ExecutionLifetimes(
            rules=tuple(LifetimeRule(target, barrier=True) for target in barriers),
            select=select,
        ),
    )
    changed = region_artifact(captured, policy, stage_id="block", variant="recompute")
    building = False
    assert calls
    assert changed.storage_contract == captured.storage_contract
    info = changed.regions.identity["lifetimes"]
    assert info["schema"] == 2 and info["views"]
    assert any(d["applied"] for d in info["compaction"]["decisions"])
    assert all(
        after <= before for _, before, after in info["compaction"]["barrier_bytes"]
    )
    torch.testing.assert_close(region_executable(changed)(*inputs)[0], layer(*inputs))
    json.dumps(changed.regions.lifetime_report)


def test_layout_contract_versions_and_compact_conflicts_are_public():
    from tests.shadowspill.pytorch.chunking._lifetime_ops import logical_consumer

    x = torch.randn(7, 12)
    captured = GraphArtifact.capture(
        kind="inference",
        graph_module=make_fx(lambda x: logical_consumer((x * 2)[:, :4], x[:, :4]))(x),
        example_inputs=(x,),
    )
    target = "lifetime_copy_test.logical_consumer.default"
    lifetimes = dict(
        rules=(
            LifetimeRule("aten.slice.Tensor", compact=True),
            LifetimeRule(target, barrier=True),
        ),
        readonly=(ReadonlyOperator(target, "effects.v1"),),
    )
    variants = [
        region_artifact(
            captured,
            ChunkRegions(
                IsolateOperations((target,), base="whole"),
                NoChunks(),
                lifetimes=ExecutionLifetimes(
                    **lifetimes,
                    layout=(LayoutIndependentOperator(target, version, (0,)),),
                ),
            ),
            stage_id="block",
            variant="recompute",
        )
        for version in ("values.v1", "values.v2")
    ]
    assert variants[0].compatibility_digest != variants[1].compatibility_digest
    for changed in variants:
        assert any(
            d["applied"]
            for d in changed.regions.identity["lifetimes"]["compaction"]["decisions"]
        )
        torch.testing.assert_close(
            region_executable(changed)(x)[0], captured.graph_module(x)
        )
    with pytest.raises(CaptureError, match="conflicting lifetime compact"):
        compose(
            captured,
            ExecutionLifetimes(
                rules=(
                    LifetimeRule("aten.slice.Tensor", compact=True),
                    LifetimeRule("aten.slice.Tensor", compact=False),
                ),
            ),
        )


@pytest.mark.parametrize("save", [False, True])
@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.parametrize("batch", [1, 2])
def test_public_primal_replay_preserves_checkpoint_cut_and_chunked_vjp(
    save, lora, batch
):
    torch.manual_seed(941)
    inputs = (
        torch.randn(batch, 17, 5, dtype=torch.float64, requires_grad=True),
        torch.randn(5, 9, dtype=torch.float64, requires_grad=not lora),
        torch.randn(5, 3, dtype=torch.float64, requires_grad=True),
        torch.randn(3, 9, dtype=torch.float64, requires_grad=True),
        torch.randn(9, 5, dtype=torch.float64, requires_grad=True),
    )

    def layer(x, weight, a, b, out):
        normalized = x.sin()
        q = normalized @ weight + (normalized @ a) @ b
        return torch.nn.functional.silu(q.softmax(1) @ out) + normalized

    expected = layer(*inputs)
    tangent = torch.randn_like(expected)
    reference = torch.autograd.grad(
        expected, [v for v in inputs if v.requires_grad], tangent
    )
    pair = capture_graph_pair(
        make_fx(layer)(*inputs),
        inputs,
        original_output=expected,
        memory_budget=float(save),
    )
    observations = []

    def select(info):
        observations.append(info)
        if info.target == "aten.sin.default":
            return LifetimeSelection(replay=True)
        return None

    policy = ChunkRegions(
        IsolateOperations(SOFTMAX, base="whole"),
        RegionChunks(17, batch),
        lifetimes=ExecutionLifetimes(
            rules=tuple(LifetimeRule(target, barrier=True) for target in SOFTMAX),
            select=select,
        ),
    )
    forward, backward = [
        region_artifact(
            artifact,
            policy,
            stage_id="block",
            variant="save" if save else "recompute",
        )
        for artifact in (pair.forward, pair.backward)
    ]
    assert forward.storage_contract == pair.forward.storage_contract
    assert backward.storage_contract == pair.backward.storage_contract
    assert any(info.phase == "primal" for info in observations)
    assert any(info.phase == "derivative" for info in observations)
    reports = [a.regions.lifetime_report.get("replay") for a in (forward, backward)]
    if not save:
        assert any(d["applied"] for r in reports if r for d in r["decisions"])
    before = len(observations)
    with torch.no_grad():
        actual = region_executable(forward)(*inputs)
        saved = actual[-pair.saved_value_count :] if pair.saved_value_count else ()
        gradients = region_executable(backward)(*saved, tangent)
    assert len(observations) == before
    torch.testing.assert_close(actual[0], expected, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(
        tuple(g for g in gradients if g is not None), reference, rtol=1e-10, atol=1e-10
    )
