"""Shared physical weights need no logical representation slicing rule."""

from dataclasses import replace

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from engine_tests.pytorch.chunking.test_composition import executable
from engine_tests.pytorch.representation_fixtures import ScaledWeight
from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.aot import capture_graph_pair
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.chunking import AxisChunks, ChunkAxis, ChunkRegions
from shadowspill.pytorch.chunking.inputs import resolve_input_axes
from shadowspill.pytorch.chunking.regions import region_artifact
from shadowspill.pytorch.chunking.transform import chunk_artifact
from shadowspill.pytorch.partition import IsolateOperations
from shadowspill.pytorch.representations import tensor_components


class Tokens:
    def configure_stage(self, stage):
        axes = {
            i: domains["tokens"]
            for i, domains in enumerate(stage.input_domains)
            if "tokens" in domains
        }
        if stage.kind == "forward":
            axes.update(
                {
                    i: ChunkAxis(0)
                    for i, geometry in enumerate(stage.inputs)
                    if geometry.shape == (7, 8)
                }
            )
        return AxisChunks(3, axes, domain="tokens") if axes else None


@pytest.mark.parametrize("save", [False, True])
@pytest.mark.parametrize("regions", [False, True])
def test_shared_wrapper_preserves_physical_payload_and_adapter_vjp(save, regions):
    torch.manual_seed(319)
    weight = ScaledWeight(
        torch.arange(-16, 16, dtype=torch.int8).reshape(4, 8), torch.tensor(0.025)
    )
    inputs = (
        torch.randn(7, 8, requires_grad=True),
        weight,
        torch.randn(8, 2, requires_grad=True),
        torch.randn(2, 4, requires_grad=True),
    )

    def layer(x, weight, a, b):
        return torch.nn.functional.silu(x @ weight.T + (x @ a) @ b)

    expected = layer(*inputs)
    tangent = torch.randn_like(expected)
    gradients = torch.autograd.grad(
        expected, (inputs[0], inputs[2], inputs[3]), tangent
    )
    pair = capture_graph_pair(
        make_fx(layer)(*inputs),
        inputs,
        original_output=expected,
        memory_budget=float(save),
    )
    assert pair.forward.input_components == (
        (0, ()),
        (1, ("payload",)),
        (1, ("scale",)),
        (2, ()),
        (3, ()),
    )
    config = AxisChunks(3, {0: 0}, domain="tokens")
    policy = ChunkRegions(IsolateOperations((), base="whole"), Tokens())
    axes, extent = resolve_input_axes(pair.forward, config)
    assert axes == (ChunkAxis(0), None, None, None, None)
    with torch.no_grad():
        if regions:
            forward = region_artifact(
                pair.forward, policy, stage_id="layer", variant="recompute"
            )
            domains = forward.regions.output_domains
            backward = region_artifact(
                pair.backward,
                policy,
                stage_id="layer",
                variant="recompute",
                input_domains=(*domains[-pair.saved_value_count :], domains[0]),
                domain_rules=forward.regions.domain_rules,
            )
            assert forward.regions.input_domains[1:3] == ({}, {})
        else:
            forward = chunk_artifact(pair.forward, axes, extent, config)
            saved = forward.chunking.outputs[-pair.saved_value_count :]
            backward = chunk_artifact(
                pair.backward, (*saved, ChunkAxis(0)), extent, config
            )
        physical = tuple(v for x in inputs for _, v in tensor_components(x))
        actual = executable(forward)(*physical)
        vjp = executable(backward)(*actual[-pair.saved_value_count :], tangent)
    torch.testing.assert_close(actual[0], expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        tuple(g for g in vjp if g is not None), gradients, rtol=1e-5, atol=1e-5
    )


def test_logical_wrapper_axis_is_still_refused():
    weight = ScaledWeight(torch.ones(4, 8, dtype=torch.int8), torch.tensor(0.25))
    inputs = weight.payload, weight.scale

    def decode(payload, scale):
        return payload.float() * scale

    artifact = replace(
        GraphArtifact.capture(
            kind="inference",
            graph_module=make_fx(decode)(*inputs),
            example_inputs=inputs,
        ),
        input_components=((0, ("payload",)), (0, ("scale",))),
    )
    with pytest.raises(CaptureError, match="invalid chunk dimension"):
        resolve_input_axes(artifact, AxisChunks(2, {0: 0}))

    class Payload:
        def configure_stage(self, stage):
            return AxisChunks(2, {0: 0}, domain="features")

    with pytest.raises(CaptureError, match="slicing them needs a representation rule"):
        region_artifact(
            artifact,
            ChunkRegions(IsolateOperations((), base="whole"), Payload()),
            stage_id="decode",
            variant=None,
        )
