"""Bounded real-provider checks for the full FP8 GLM execution contracts."""

import json
from math import ceil

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

from shadowspill.errors import CaptureError
from shadowspill.pytorch import AxisChunks, ChunkAxis
from shadowspill.pytorch.capture.aot import capture_graph_pair
from shadowspill.pytorch.chunking.transform import chunk_artifact
from tests.shadowspill.pytorch.chunking.test_axis_vjp import executable
from tests.workloads.test_sonic_axis_chunking import Calls, close
from workloads.models.shared.chunking import operator_rules


def fp8_inputs(batch=1, *, device="cpu", empty_experts=False):
    generator = torch.Generator().manual_seed(719)

    def draw(shape, dtype=torch.bfloat16, train=True):
        value = torch.randn(shape, generator=generator) * shape[-1] ** -0.5
        return value.to(device=device, dtype=dtype).requires_grad_(train)

    # Physical checkpoint layout: one E4M3 copy plus FP32 128x128 block scales.
    def weight(shape):
        value = torch.randn(shape, generator=generator) * 20
        data = value.to(device=device, dtype=torch.float8_e4m3fn)
        scales = torch.full(
            (shape[0], ceil(shape[1] / 128), ceil(shape[2] / 128)),
            1 / (20 * shape[-1] ** 0.5),
            dtype=torch.float32,
            device=device,
        )
        return data, scales

    up, up_scale = weight((8, 256, 256))
    down, down_scale = weight((8, 256, 128))
    values = (
        draw((batch, 160, 256)),
        draw((batch, 160, 256)),
        draw((8, 256), torch.float32),
        up,
        down,
        draw((8, 32, 256)),
        draw((8, 256, 32)),
        draw((8, 32, 128)),
        draw((8, 256, 32)),
        up_scale,
        down_scale,
    )
    if empty_experts:
        with torch.no_grad():
            values[0].abs_().add_(0.25)
            values[2].fill_(-0.05)
            values[2][0].fill_(0.05)
            values[2][1].fill_(0.025)
    return values


def fp8_region(*values, builtin=False):
    from mlops.block_fp8 import BlockFP8Weight

    args = list(values[:9])
    for position, scale in ((3, values[9]), (4, values[10])):
        args[position] = BlockFP8Weight(args[position], scale)
    return wrapped_fp8_region(*args, builtin=builtin)


def wrapped_fp8_region(*args, builtin=False):
    from mlops.providers.sonicmoe.lora import expert_lora

    if builtin:
        from mlops.lora.experts import expert_lora
    output, _, counts, probabilities = expert_lora(
        *args,
        top_k=2,
        routing_mode="sigmoid_noaux_tc",
        scale=0.5,
        **(
            {"implementation": "builtin", "compute_precision": "bf16"}
            if builtin
            else {}
        ),
    )
    return output, counts, probabilities


def captured(function, values, size, axes, rules, tangent_axes):
    with FakeTensorMode(allow_non_fake_inputs=True) as mode:
        inputs = tuple(mode.from_tensor(value) for value in values)
        pair = capture_graph_pair(
            make_fx(function)(*inputs),
            inputs,
            original_output=function(*inputs),
            memory_budget=0.0,
        )
        config = AxisChunks(
            size, {i: axis for i, axis in enumerate(axes) if axis}, rules=rules
        )
        forward = chunk_artifact(pair.forward, axes, 160, config)
        saved = forward.chunking.outputs[-pair.saved_value_count :]
        backward = chunk_artifact(pair.backward, (*saved, *tangent_axes), 160, config)
    return pair, forward, backward


def fp8_capture(values, batch, size):
    from mlops.integrations.shadowspill import sonicmoe_chunk_rules

    axis = ChunkAxis(1)
    return captured(
        fp8_region,
        values,
        size,
        (axis, axis, *(None,) * 9),
        (*operator_rules(batch), *sonicmoe_chunk_rules(compute_precision="fp8_block")),
        (axis, None),
    )


def test_block_fp8_capture_keeps_scales_shared_and_routes_retained(monkeypatch):
    # Avoid Quack's default device discovery. Other provider imports may still
    # inspect a visible device; this checks graph structure, not CPU-only build.
    monkeypatch.setenv("QUACK_ARCH", "90")
    pair, forward, backward = fp8_capture(fp8_inputs(), 1, 64)
    assert "mlops.lora_route.default" not in pair.backward.operator_targets
    assert "mlops.lora_route.default" not in pair.retention.regenerated_operators
    assert forward.chunking.inputs[-2:] == (None, None)
    for artifact in (forward, backward):
        assert {
            size
            for size, _ in (
                *artifact.chunking.children,
                *artifact.chunking.accumulating_children,
            )
        } == {32, 64}


def test_block_fp8_nested_recipe_regions_capture_both_variants(monkeypatch):
    from mlops.block_fp8 import BlockFP8Weight

    from shadowspill.pytorch.chunking.regions import region_artifact
    from workloads.recipes.text.glm53_flash_lora.policies import build_options

    monkeypatch.setenv("QUACK_ARCH", "90")
    options = build_options(
        dict(sequence_length=160, chunk_size=64, moe_chunk_size=128), 1
    )
    with FakeTensorMode(allow_non_fake_inputs=True) as mode:
        raw = tuple(mode.from_tensor(value) for value in fp8_inputs())
        inputs = list(raw[:9])
        inputs[3] = BlockFP8Weight(raw[3], raw[9])
        inputs[4] = BlockFP8Weight(raw[4], raw[10])
        inputs = tuple(inputs)
        for variant in ("save", "recompute"):
            pair = capture_graph_pair(
                make_fx(wrapped_fp8_region)(*inputs),
                inputs,
                original_output=wrapped_fp8_region(*inputs),
                retention=getattr(options["graph_pair_policy"], variant),
            )
            forward = region_artifact(
                pair.forward, options["chunking"], stage_id="layer", variant=variant
            )
            assert sum(bool(path) for _, path in forward.input_components) == 4
            assert all(
                not domain
                for (_, path), domain in zip(
                    forward.input_components,
                    forward.regions.input_domains,
                    strict=True,
                )
                if path
            )
            domains = forward.regions.output_domains
            backward = region_artifact(
                pair.backward,
                options["chunking"],
                stage_id="layer",
                variant=variant,
                input_domains=(
                    *domains[-pair.saved_value_count :],
                    domains[0],
                    domains[2],
                ),
                domain_rules=forward.regions.domain_rules,
            )
            for artifact in (forward, backward):
                outer = artifact.regions.children[0].chunking
                assert outer is not None
                assert outer.identity["chunk_size"] == 128
                assert all(child.regions for _, child in outer.children)


def require_hopper():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("requires the H100 FP8 and fused mHC providers")


@pytest.mark.cuda
@pytest.mark.slow
@pytest.mark.parametrize("batch,size,empty", [(1, 64, False), (2, 96, True)])
def test_block_fp8_chunks_preserve_outputs_and_arbitrary_vjp(batch, size, empty):
    require_hopper()
    from mlops.dispatch.context import weight_gradients_at

    values = fp8_inputs(batch, device="cuda", empty_experts=empty)
    with weight_gradients_at(torch.float32):
        pair, forward, backward = fp8_capture(values, batch, size)
        ordinary = fp8_region(*values)
        torch.manual_seed(918)
        tangents = torch.randn_like(ordinary[0]), torch.randn_like(ordinary[2])
        parameters = [v for v in values if v.requires_grad]
        eager = torch.autograd.grad((ordinary[0], ordinary[2]), parameters, tangents)
        independent = fp8_region(*values, builtin=True)
        oracle = torch.autograd.grad(
            (independent[0], independent[2]), parameters, tangents
        )
        with torch.no_grad(), Calls() as calls:
            actual = executable(forward)(*values)
            forward_calls = calls.counts.copy()
            calls.counts.clear()
            gradients = executable(backward)(
                *actual[-pair.saved_value_count :], *tangents
            )
            backward_calls = calls.counts.copy()
    chunks = ceil(160 / size)
    assert forward_calls["mlops.lora_route.default"] == chunks
    assert backward_calls["mlops.lora_route.default"] == 0
    for name in ("up", "down"):
        assert backward_calls[f"mlops.lora_sonicmoe_{name}_fwd.default"] == chunks
        assert backward_calls[f"mlops.lora_sonicmoe_{name}_bwd.default"] == chunks
    torch.testing.assert_close(actual[1], ordinary[1], rtol=0, atol=0)
    torch.testing.assert_close(actual[2], ordinary[2], rtol=2e-6, atol=2e-6)
    if empty:
        assert actual[1][2:].count_nonzero() == 0
        assert all(g[2:].count_nonzero() == 0 for g in gradients[5:9])
    tested = (actual[0], *(g for g in gradients if g is not None))
    labels = (
        "output",
        "hidden",
        "residual",
        "router",
        "up_a",
        "up_b",
        "down_a",
        "down_b",
    )
    errors = {}
    for name, expected, bound in (
        ("same_fp8_provider", (ordinary[0], *eager), 0.03),
        ("independent_dequantized_bf16", (independent[0], *oracle), 0.10),
    ):
        errors[name] = {
            label: close(a, b, bound=bound, name=label)
            for label, a, b in zip(labels, tested, expected, strict=True)
        }
    print(
        json.dumps(dict(batch=batch, chunk=size, empty=empty, errors=errors)),
        flush=True,
    )


def mhc_inputs(*, device="cpu"):
    torch.manual_seed(171)
    return (
        torch.randn(160, 4, 128, device=device, dtype=torch.bfloat16).requires_grad_(),
        (torch.randn(24, 512, device=device) * 0.02).requires_grad_(),
        (torch.randn(24, device=device) * 0.01).requires_grad_(),
        torch.full((3,), 0.1, device=device, requires_grad=True),
    )


def mhc_region(streams, projection, bias, scale):
    from mlops.providers.deepgemm.hyper_connection import _forward
    from mlops.providers.tilekernels.hyper_connection import _post

    post, mixing, branch, _, _ = _forward(
        streams, projection, bias, scale, 1e-5, 1e-6, 20, 1
    )
    # A token-local nonlinear branch gives independent pre/post/input VJPs.
    return _post(
        branch.sin().view(1, -1, 128),
        streams.view(1, -1, 4, 128),
        post.view(1, -1, 4, 1),
        mixing.view(1, -1, 4, 4),
    ).view_as(streams)


def mhc_capture(values, size):
    from mlops.integrations.shadowspill import mhc_chunk_rules

    return captured(
        mhc_region,
        values,
        size,
        (ChunkAxis(0), None, None, None),
        mhc_chunk_rules(),
        (ChunkAxis(0),),
    )


def test_mhc_capture_checks_tail_alignment_before_kernel_execution():
    initialized = torch.cuda.is_initialized()
    _, forward, backward = mhc_capture(mhc_inputs(), 64)
    assert {
        n
        for n, _ in (
            *backward.chunking.children,
            *backward.chunking.accumulating_children,
        )
    } == {32, 64}
    assert forward.chunking.rows == 160
    with pytest.raises((CaptureError, ValueError), match="divisible by 32"):
        mhc_capture(mhc_inputs(), 48)
    assert torch.cuda.is_initialized() == initialized


@pytest.mark.cuda
@pytest.mark.slow
def test_fused_mhc_chunks_preserve_parameter_reductions_and_vjp():
    require_hopper()
    values = mhc_inputs(device="cuda")
    pair, forward, backward = mhc_capture(values, 64)
    ordinary = mhc_region(*values)
    tangent = torch.randn_like(ordinary)
    expected = torch.autograd.grad(ordinary, values, tangent)
    with torch.no_grad():
        actual = executable(forward)(*values)
        gradients = executable(backward)(*actual[-pair.saved_value_count :], tangent)
    errors = {
        name: close(a, b, bound=0.02, name=name)
        for name, a, b in zip(
            ("output", "streams", "projection", "bias", "scale"),
            (actual[0], *gradients),
            (ordinary, *expected),
            strict=True,
        )
    }
    print(json.dumps(dict(mhc_errors=errors)), flush=True)
