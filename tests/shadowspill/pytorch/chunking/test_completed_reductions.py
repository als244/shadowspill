"""Whole regions may aggregate values produced by completed chunk loops."""

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from shadowspill.pytorch.capture.aot import capture_graph_pair
from shadowspill.pytorch.chunking import ChunkRegions
from shadowspill.pytorch.chunking.regions import region_artifact
from shadowspill.pytorch.partition import IsolateOperations
from tests.shadowspill.pytorch.chunking.test_regions import (
    RegionChunks,
    region_executable,
)

WHOLE = (
    "aten.stack.default",
    "aten.tanh.default",
    "aten.tanh_backward.default",
    "aten.sort.default",
    "aten.scatter.src",
)


class AggregateChunks(RegionChunks):
    def configure_stage(self, stage):
        if any(target in WHOLE for target in stage.operators):
            return None
        return super().configure_stage(stage)


@pytest.mark.parametrize("aggregation", ["mixed", "nonlinear", "tuple"])
@pytest.mark.parametrize("save", [False, True])
def test_whole_aggregation_consumes_completed_chunk_reductions(aggregation, save):
    torch.manual_seed(731)
    x = torch.randn(29, 5, dtype=torch.float64, requires_grad=True)
    shared = torch.randn(5, dtype=torch.float64, requires_grad=True)

    def layer(x, shared):
        total = x.sin().sum(0)
        if aggregation == "mixed":
            scale = torch.stack((total, shared)).sum(0)
        elif aggregation == "nonlinear":
            scale = total.tanh() * shared
        else:
            scale = total.sort().values * shared
        return scale

    expected = layer(x, shared)
    tangent = torch.randn_like(expected)
    reference = torch.autograd.grad(expected, (x, shared), tangent)
    pair = capture_graph_pair(
        make_fx(layer)(x, shared),
        (x, shared),
        original_output=layer(x, shared),
        memory_budget=float(save),
    )
    policy = ChunkRegions(IsolateOperations(WHOLE, base="whole"), AggregateChunks(29))
    variant = "save" if save else "recompute"
    forward, backward = (
        region_artifact(
            artifact,
            policy,
            stage_id="aggregate",
            variant=variant,
            gradient_leaves=(1,),
        )
        for artifact in (pair.forward, pair.backward)
    )
    assert any(child.chunking for child in forward.regions.children)
    assert any(child.chunking for child in backward.regions.children)
    with torch.no_grad():
        actual = region_executable(forward)(x, shared)
        saved = actual[-pair.saved_value_count :] if pair.saved_value_count else ()
        gradients = region_executable(backward)(*saved, tangent)
    torch.testing.assert_close(actual[0], expected, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(gradients, reference, rtol=1e-10, atol=1e-10)
