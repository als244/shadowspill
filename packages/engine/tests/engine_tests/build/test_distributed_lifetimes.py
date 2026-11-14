"""CPU preflight of the actual DP lifetime fixture and its independent VJP."""

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from engine_tests.execution._distributed_regions import (
    Block,
    Network,
    check_batch_chunk_metadata,
    check_lifetime_reports,
    policies,
)
from engine_tests.pytorch.chunking.test_composition import executable
from shadowspill.pytorch.capture.aot import capture_graph_pair
from shadowspill.pytorch.chunking.regions import region_artifact


@pytest.mark.parametrize(
    "mode,lora,nested,retained,chunk_axis,batch",
    [
        ("move", False, False, False, "sequence", 2),
        ("replay_compact", False, True, False, "sequence", 2),
        ("replay_compact", True, False, True, "sequence", 2),
        ("replay_compact", True, True, True, "sequence", 2),
        ("replay_compact", False, True, True, "batch", 13),
        ("replay_compact", True, True, False, "batch", 13),
    ],
)
def test_dp_fixture_lifetimes_preserve_checkpoint_cut_and_arbitrary_vjp(
    mode, lora, nested, retained, chunk_axis, batch
):
    torch.manual_seed(731)
    block = Block(lora, dtype=torch.float64, lifetime_mode=mode)
    weights = dict(block.named_parameters())
    inputs = (
        torch.randn(batch, 17, 7, dtype=torch.float64, requires_grad=True),
        *weights.values(),
    )

    def layer(x, *values):
        return torch.func.functional_call(
            block, dict(zip(weights, values, strict=True)), (x,)
        )

    expected = layer(*inputs)
    tangent = torch.randn_like(expected)
    gradients = torch.autograd.grad(
        expected, [value for value in inputs if value.requires_grad], tangent
    )
    controls = policies(
        dict(
            storage="sharded",
            checkpoint_cuts=False,
            nested=nested,
            retain_sequence=retained,
            lifetime_mode=mode,
            chunk_axis=chunk_axis,
            batch=batch,
        )
    )
    pair = capture_graph_pair(
        make_fx(layer)(*inputs),
        inputs,
        original_output=layer(*inputs),
        memory_budget=0,
        retention=controls["graph_pair_policy"].recompute,
    )
    forward, backward = [
        region_artifact(
            artifact,
            controls["chunking"],
            stage_id="block",
            variant="recompute",
        )
        for artifact in (pair.forward, pair.backward)
    ]
    assert forward.storage_contract == pair.forward.storage_contract
    assert backward.storage_contract == pair.backward.storage_contract
    if chunk_axis == "batch":
        check_batch_chunk_metadata(
            [artifact.regions.identity for artifact in (forward, backward)]
        )
    check_lifetime_reports(
        [artifact.regions.lifetime_report for artifact in (forward, backward)], mode
    )
    selector = controls["chunking"].lifetimes.select
    assert selector.calls > 0
    selector.building = False
    before = selector.calls
    with torch.no_grad():
        actual = executable(forward)(*inputs)
        saved = actual[-pair.saved_value_count :] if pair.saved_value_count else ()
        actual_gradients = executable(backward)(*saved, tangent)
    assert selector.calls == before
    torch.testing.assert_close(actual[0], expected, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(
        tuple(value for value in actual_gradients if value is not None),
        gradients,
        rtol=1e-10,
        atol=1e-10,
    )
    # The two-block DP fixture preserves tied, nonzero factors and odd shard sizes.
    network = Network(lora, lifetime_mode=mode)
    assert network.blocks[0].out is network.blocks[1].out
    assert network.blocks[0].b is network.blocks[1].b
    assert network.blocks[0].out.numel() % 2 == 1
    assert network.blocks[0].b.count_nonzero() == network.blocks[0].b.numel()
    assert network.blocks[0].weight.requires_grad is not lora
