"""Logical axes survive a whole region before a later chunked consumer."""

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.chunking import (
    AxisChunks,
    AxisFrom,
    ChunkAxis,
    ChunkRegions,
    OperatorRule,
)
from shadowspill.pytorch.chunking.regions import region_artifact
from tests.shadowspill.pytorch.chunking.test_regions import region_executable


@torch.library.custom_op("shadowspill_test::apply_flat_statistics", mutates_args=())
def apply_flat_statistics(x: torch.Tensor, statistics: torch.Tensor) -> torch.Tensor:
    # Mirrors a provider that indexes its flat auxiliary state using x's extent.
    # Passing the entire state's prefix for every chunk gives wrong values.
    scale = statistics[: x.numel() // x.shape[-1]].reshape(*x.shape[:-1], 1)
    return x * scale


@apply_flat_statistics.register_fake
def _fake_apply(x, statistics):
    return torch.empty_like(x)


TARGET = "shadowspill_test.apply_flat_statistics.default"
RULE = OperatorRule(TARGET, "1", (AxisFrom(0),))


@torch.library.custom_op("shadowspill_test::flat_statistics", mutates_args=())
def flat_statistics(x: torch.Tensor) -> torch.Tensor:
    return (x.square().mean(-1) + 0.25).rsqrt().reshape(-1)


@flat_statistics.register_fake
def _fake_statistics(x):
    return x.new_empty((x.numel() // x.shape[-1],))


STATISTICS_RULE = OperatorRule(
    "shadowspill_test.flat_statistics.default", "1", (AxisFrom(0, 1),)
)


class DeclaredStatistics:
    def configure_stage(self, stage):
        return AxisChunks(
            4, {0: ChunkAxis(1)}, rules=(STATISTICS_RULE,), domain="tokens"
        )


class Whole:
    def configure_stage(self, stage):
        return None


class SplitStatistics:
    def assign_stages(self, graph_module, module):
        return {
            node.name: int(str(node.target) == TARGET)
            for node in graph_module.graph.nodes
            if node.op == "call_function"
        }


class ConsumeStatistics:
    def __init__(self, sequence):
        self.sequence = sequence

    def configure_stage(self, stage):
        if TARGET not in stage.operators:
            return None
        axes = {}
        for index, value in enumerate(stage.inputs):
            if stage.input_domains and "tokens" in stage.input_domains[index]:
                axes[index] = stage.input_domains[index]["tokens"]
            elif len(value.shape) > 1 and value.shape[1] == self.sequence:
                axes[index] = ChunkAxis(1)
        return AxisChunks(4, axes, rules=(RULE,), domain="tokens")


@pytest.mark.parametrize("batch,sequence,heads", [(1, 13, 2), (2, 17, 3)])
@pytest.mark.parametrize("opaque", [False, True])
def test_flat_statistics_from_whole_region_are_sliced_with_their_tokens(
    batch, sequence, heads, opaque
):
    torch.manual_seed(731)
    x = torch.randn(batch, sequence, heads, 5, dtype=torch.float64)

    def region(x):
        statistics = (
            flat_statistics(x)
            if opaque
            else (x.square().mean(-1) + 0.25).rsqrt().reshape(-1)
        )
        return (apply_flat_statistics(x, statistics),)

    forward_artifact = GraphArtifact.capture(
        kind="forward",
        graph_module=make_fx(lambda x: (flat_statistics(x),))(x),
        example_inputs=(x,),
    )
    forward = region_artifact(
        forward_artifact,
        ChunkRegions("whole", DeclaredStatistics()),
        stage_id="norm_forward",
        variant="recompute",
    )
    assert forward.regions is not None
    artifact = GraphArtifact.capture(
        kind="backward", graph_module=make_fx(region)(x), example_inputs=(x,)
    )
    changed = region_artifact(
        artifact,
        ChunkRegions(SplitStatistics(), ConsumeStatistics(sequence)),
        stage_id="norm_backward",
        variant="recompute",
        input_domains=forward.regions.input_domains,
        domain_rules=forward.regions.domain_rules,
    )
    actual = region_executable(changed)(x)
    expected = x * (x.square().mean(-1) + 0.25).rsqrt().unsqueeze(-1)
    torch.testing.assert_close(actual[0], expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("certified", [False, True])
def test_whole_provider_needs_declared_contract_to_propagate_domains(certified):
    x = torch.randn(2, 17, 3, 5, dtype=torch.float64)
    artifact = GraphArtifact.capture(
        kind="forward",
        graph_module=make_fx(lambda x: (flat_statistics(x),))(x),
        example_inputs=(x,),
    )
    rules = (
        {name: (STATISTICS_RULE,) for name in ("tokens", "heads")} if certified else {}
    )
    changed = region_artifact(
        artifact,
        ChunkRegions("whole", Whole()),
        stage_id="whole",
        variant="recompute",
        input_domains=({"tokens": ChunkAxis(1), "heads": ChunkAxis(2)},),
        domain_rules=rules,
    )
    assert changed.regions is not None
    assert all(child.chunking is None for child in changed.regions.children)
    assert changed.regions.output_domains == (
        {"tokens": ChunkAxis(0, before=2, after=3), "heads": ChunkAxis(0, before=34)}
        if certified
        else {},
    )


def test_invalid_declared_whole_provider_mapping_is_not_silently_discarded():
    x = torch.randn(2, 17, 3, 5, dtype=torch.float64)
    artifact = GraphArtifact.capture(
        kind="forward",
        graph_module=make_fx(lambda x: (flat_statistics(x),))(x),
        example_inputs=(x,),
    )
    incorrect = OperatorRule(STATISTICS_RULE.target, "invalid", (AxisFrom(0, 2),))
    with pytest.raises(CaptureError):
        region_artifact(
            artifact,
            ChunkRegions("whole", Whole()),
            stage_id="whole",
            variant="recompute",
            input_domains=({"tokens": ChunkAxis(1)},),
            domain_rules={"tokens": (incorrect,)},
        )


def test_conflicting_forward_and_backward_domain_contracts_are_refused():
    x = torch.randn(1, 13, 2, 5, dtype=torch.float64)
    artifact = GraphArtifact.capture(
        kind="backward",
        graph_module=make_fx(lambda x: (flat_statistics(x),))(x),
        example_inputs=(x,),
    )
    previous = OperatorRule(STATISTICS_RULE.target, "previous", (AxisFrom(0, 1),))
    with pytest.raises(CaptureError, match="conflicting operator contracts"):
        region_artifact(
            artifact,
            ChunkRegions("whole", DeclaredStatistics()),
            stage_id="backward",
            variant="recompute",
            input_domains=({"tokens": ChunkAxis(1)},),
            domain_rules={"tokens": (previous,)},
        )
