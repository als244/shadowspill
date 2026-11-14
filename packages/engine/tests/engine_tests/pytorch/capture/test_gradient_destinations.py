"""Grouped custom destinations preserve VJPs, ownership and cached graphs."""

from dataclasses import replace

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from engine_tests.pytorch.capture._gradient_destination_ops import (
    CALLS,
    function,
    interleave,
)
from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.capture.gradients import (
    DestinationView,
    GradientDestinationMapping,
    GradientDestinationRule,
    accumulate_gradient_outputs,
)
from shadowspill.pytorch.capture.gradients.destinations import (
    fuse_gradient_destinations,
)
from shadowspill.pytorch.graph_pairs.serialization import CachedGraphArtifact
from shadowspill.task.storage import StorageRootKind


def canonical_mapping(context):
    # This provider understands the exact interleave/scale path, without an
    # inverse view or a full raw B-gradient buffer. Other paths decline.
    selected = []
    canonical_b = False
    for item in context.destinations:
        if item.output_index == 1 and not item.operations:
            selected.append(1)
        elif item.output_index == 2:
            targets = tuple(node.target for node in item.operations)
            if not targets:
                selected.append(2)
            if (
                targets
                == (
                    torch.ops.aten.mul.Tensor,
                    torch.ops.aten._unsafe_view.default,
                    torch.ops.aten.clone.default,
                    torch.ops.aten.transpose.int,
                    torch.ops.aten.view.default,
                )
                and item.operations[0].args[1] == context.producer.args[4]
            ):
                selected.append(2)
                canonical_b = True
    return (
        GradientDestinationMapping(
            tuple(selected), ((),) * len(selected), {"canonical_b": canonical_b}
        )
        if selected
        else None
    )


RULE = GradientDestinationRule(
    "gradient_destination_test.fresh.default",
    "gradient_destination_test.accumulating.default",
    (1, 2),
    1,
    canonical_mapping,
)


def artifact(inputs, function=function):
    graph = make_fx(function, tracing_mode="fake")(*inputs)
    return GraphArtifact.capture(
        kind="backward", graph_module=graph, example_inputs=inputs
    )


def values(rows=11, dtype=torch.float32):
    generator = torch.Generator().manual_seed(7301)
    return tuple(
        torch.randn(shape, generator=generator, dtype=dtype)
        for shape in ((rows, 7), (3, 7), (10, 3), (rows, 10))
    )


def reference(inputs):
    x, a, b, tangent = inputs
    x, a, b = [v.detach().clone().requires_grad_(True) for v in (x, a, b)]
    return torch.autograd.grad(((x @ a.T) @ interleave(b).T) * 1.75, (x, a, b), tangent)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_grouped_custom_destinations_across_chunks_tails_and_microbatches(dtype):
    x, a, b, tangent = values(dtype=dtype)
    expected = [torch.zeros_like(a), torch.zeros_like(b)]
    priors, pointers = None, None
    CALLS.clear()
    for micro in range(2):
        current = (x + micro * 0.25, a, b, tangent - micro * 0.5)
        dx, da, db = reference(current)
        expected[0].add_(da)
        expected[1].add_(db)
        pieces = []
        for start in range(0, x.shape[0], 4):
            inputs = (
                current[0][start : start + 4],
                a,
                b,
                current[3][start : start + 4],
            )
            fresh_graph = artifact(inputs)
            if priors is None:
                result = fresh_graph.graph_module(*inputs)
                priors = result[1:]
                pointers = [v.data_ptr() for v in priors]
            else:
                graph = accumulate_gradient_outputs(
                    fresh_graph, (1, 2), gradient_destination_rules=(RULE,)
                )
                assert graph.operator_targets.count(RULE.replacement) == 1
                assert RULE.target not in graph.operator_targets
                assert "aten.add_.Tensor" not in graph.operator_targets
                contract = graph.storage_contract
                assert {m.input_position for m in contract.mutations} == {4, 5}
                assert sum(r.kind is StorageRootKind.FRESH for r in contract.roots) == 1
                restored = CachedGraphArtifact.capture(graph).restore()
                assert restored.compatibility_digest == graph.compatibility_digest
                assert restored.storage_contract == contract
                result = restored.graph_module(*inputs, *priors)
                assert [v.data_ptr() for v in result[1:]] == pointers
            pieces.append(result[0])
        tolerance = 2e-5 if dtype is torch.float32 else 1e-11
        for actual, wanted in zip(
            (torch.cat(pieces), *priors), (dx, *expected), strict=True
        ):
            torch.testing.assert_close(actual, wanted, rtol=tolerance, atol=tolerance)
    assert CALLS == {"calls": 6, "fresh_parameter_buffers": 2}


@pytest.mark.parametrize("leaves", [(1,), (2,), (1, 2)])
def test_partial_destinations_preserve_other_outputs_and_version_identity(leaves):
    inputs = values()
    plain = artifact(inputs)
    expected = reference(inputs)
    result = accumulate_gradient_outputs(
        plain, leaves, gradient_destination_rules=(RULE,)
    )
    changed = accumulate_gradient_outputs(
        plain, leaves, gradient_destination_rules=(replace(RULE, version=2),)
    )
    assert result.compatibility_digest != changed.compatibility_digest
    priors = [torch.ones_like(expected[i]) for i in leaves]
    CALLS.clear()
    actual = result.graph_module(*inputs, *priors)
    for index, value in enumerate(actual):
        torch.testing.assert_close(
            value, expected[index] + int(index in leaves), rtol=2e-5, atol=2e-5
        )
    assert CALLS["calls"] == 1
    assert CALLS["fresh_parameter_buffers"] == 2 - len(leaves)


def test_declarative_mapping_fuses_supported_output_and_keeps_nonview_fallback():
    inputs = values()
    graph = accumulate_gradient_outputs(
        artifact(inputs),
        (1, 2),
        gradient_destination_rules=(replace(RULE, mapper=None),),
    )
    assert graph.operator_targets.count(RULE.replacement) == 1
    assert "aten.add_.Tensor" in graph.operator_targets
    expected = reference(inputs)
    actual = graph.graph_module(
        *inputs, torch.ones_like(inputs[1]), torch.ones_like(inputs[2])
    )
    for index, value in enumerate(actual):
        torch.testing.assert_close(
            value, expected[index] + int(index > 0), rtol=2e-5, atol=2e-5
        )


def test_mapper_cannot_pass_a_copy_as_a_writable_destination_view():
    def copied(context):
        return GradientDestinationMapping(
            (1,), ((DestinationView(torch.ops.aten.clone.default),),)
        )

    with pytest.raises(CaptureError, match="preserve destination storage"):
        accumulate_gradient_outputs(
            artifact(values()),
            (1,),
            gradient_destination_rules=(replace(RULE, mapper=copied),),
        )


def test_mapper_cannot_write_an_overlapping_destination_view():
    def overlap(context):
        return GradientDestinationMapping(
            (1,),
            ((DestinationView(torch.ops.aten.as_strided.default, ([3, 7], [0, 1])),),),
        )

    with pytest.raises(CaptureError, match="overlapping elements"):
        accumulate_gradient_outputs(
            artifact(values()),
            (1,),
            gradient_destination_rules=(replace(RULE, mapper=overlap),),
        )


def test_externally_observed_output_is_not_consumed():
    def shared(*inputs):
        dx, da, db = function(*inputs)
        return dx, da, db, da * 2

    inputs = values()
    graph = accumulate_gradient_outputs(
        artifact(inputs, shared), (1, 2), gradient_destination_rules=(RULE,)
    )
    expected = reference(inputs)
    actual = graph.graph_module(
        *inputs, torch.ones_like(inputs[1]), torch.ones_like(inputs[2])
    )
    torch.testing.assert_close(actual[1], expected[1] + 1, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(actual[3], expected[1] * 2, rtol=2e-5, atol=2e-5)
    assert "aten.add_.Tensor" in graph.operator_targets


def test_repeated_leaf_and_repeated_rules_refuse_ambiguous_destinations():
    graph = artifact(values())
    with pytest.raises(CaptureError, match="leaves must be distinct"):
        accumulate_gradient_outputs(graph, (1, 1))
    with pytest.raises(CaptureError, match="repeat an operator"):
        accumulate_gradient_outputs(
            graph, (1, 2), gradient_destination_rules=(RULE, RULE)
        )


@torch.library.custom_op("gradient_destination_test::single", mutates_args=())
def single(x: torch.Tensor) -> torch.Tensor:
    return x.T @ x


@single.register_fake
def _single_fake(x):
    return x.new_empty((x.shape[1], x.shape[1]))


@torch.library.custom_op(
    "gradient_destination_test::single_accumulating",
    mutates_args=("destinations",),
    schema="(Tensor x, Tensor(a!)[] destinations, int[] output_indices, "
    "int contract_version) -> Tensor?",
)
def single_accumulating(
    x: torch.Tensor,
    destinations: list[torch.Tensor],
    output_indices: list[int],
    contract_version: int,
) -> torch.Tensor | None:
    assert output_indices == [0] and contract_version == 1
    torch.addmm(destinations[0], x.T, x, out=destinations[0])
    return None


@single_accumulating.register_fake
def _single_accumulating_fake(x, destinations, output_indices, contract_version):
    return None


def test_single_tensor_producer_keeps_mutating_call_and_returns_prior():
    x = values()[0]
    rule = GradientDestinationRule(
        "gradient_destination_test.single.default",
        "gradient_destination_test.single_accumulating.default",
        (0,),
        1,
    )
    graph = artifact((x,), lambda x: (single(x),))
    changed = accumulate_gradient_outputs(
        graph, (0,), gradient_destination_rules=(rule,)
    )
    prior = torch.ones(x.shape[1], x.shape[1])
    result = changed.graph_module(x, prior)[0]
    assert result.data_ptr() == prior.data_ptr()
    torch.testing.assert_close(result, x.T @ x + 1)
    assert changed.operator_targets == (rule.replacement,)


def test_aliased_destinations_are_not_consumed():
    inputs = values()
    backing = torch.zeros(64)
    priors = (backing[:21].view(3, 7), backing[:30].view(10, 3))
    captured = artifact((*inputs, *priors), lambda *args: function(*args[:4]))
    graph = captured.graph_module.graph
    placeholders = [node for node in graph.nodes if node.op == "placeholder"]
    output = next(node for node in graph.nodes if node.op == "output")
    leaves = output.args[0]
    consumed, dead = fuse_gradient_destinations(
        graph,
        [(leaves[1], placeholders[-2]), (leaves[2], placeholders[-1])],
        (RULE,),
        outputs=leaves,
        fake_mode=None,
        fuse_narrow_gradient_accumulation=True,
    )
    assert not consumed and not dead
    assert not any(
        node.target is torch.ops.gradient_destination_test.accumulating.default
        for node in graph.nodes
    )


def test_replacement_must_declare_destination_mutation():
    with pytest.raises(CaptureError, match="declare destinations mutable"):
        accumulate_gradient_outputs(
            artifact(values()),
            (1,),
            gradient_destination_rules=(replace(RULE, replacement=RULE.target),),
        )


def test_narrow_default_mapper_respects_fusion_permission():
    inputs = (values(dtype=torch.bfloat16)[0],)
    rule = GradientDestinationRule(
        "gradient_destination_test.single.default",
        "gradient_destination_test.single_accumulating.default",
        (0,),
        1,
    )
    base = artifact(inputs, lambda x: (single(x),))
    strict = accumulate_gradient_outputs(
        base,
        (0,),
        fuse_narrow_gradient_accumulation=False,
        gradient_destination_rules=(rule,),
    )
    fused = accumulate_gradient_outputs(base, (0,), gradient_destination_rules=(rule,))
    assert rule.target in strict.operator_targets
    assert rule.replacement not in strict.operator_targets
    assert rule.replacement in fused.operator_targets
    prior = torch.randn(7, 7).bfloat16()
    expected = prior + single(*inputs)
    result = strict.graph_module(*inputs, prior.clone())[0]
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
