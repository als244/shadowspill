"""Provider row contracts preserve routed values, statistics and explicit VJPs."""

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from shadowspill.integrations.mlops import operator_chunk_rules
from shadowspill.pytorch.capture.aot import capture_graph_pair
from shadowspill.pytorch.capture.artifacts import GraphArtifact, TensorGeometry
from shadowspill.pytorch.chunking import AxisChunks, ChunkAxis, StageInfo
from shadowspill.pytorch.chunking.transform import chunk_artifact
from shadowspill.task.inputs import TaskInputRole
from shadowspill_workloads.models.shared.chunking import TextAxisChunks, is_barrier
from tests.shadowspill.pytorch.chunking.test_axis_vjp import executable


@pytest.mark.parametrize("provider", ["fla", "fla_ieee", "cula"])
@pytest.mark.parametrize("direction", ["fwd", "bwd"])
def test_kimi_recurrence_is_a_complete_sequence_boundary(provider, direction):
    assert is_barrier(f"mlops.kimi_delta_attention_{provider}_{direction}.default")


@pytest.mark.parametrize("provider", ["builtin_grouped_gemm", "sonicmoe"])
def test_moe_with_embedded_sequence_lengths_stays_whole(provider):
    for entry in (
        "fwd",
        "bwd",
        "prepare_fwd",
        "finish_fwd",
        "prepare_bwd",
        "finish_bwd",
    ):
        assert is_barrier(f"mlops.moe_{provider}_{entry}.default")
    assert not is_barrier("mlops.lora_grouped_linear_fwd.default")


@pytest.mark.cuda
@pytest.mark.parametrize("budget", [0.0, 1.0])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("precision", ["fp32", "activation"])
def test_mhc_stream_expansion_preserves_token_coordinates_and_vjps(
    budget, batch, precision
):
    from mlops import mhc_combine
    from mlops.dispatch import deterministic_kernels

    torch.manual_seed(731)
    inputs = tuple(
        torch.randn(*shape, device="cuda", dtype=dtype).requires_grad_()
        for shape, dtype in (
            ((batch * 37, 64), torch.bfloat16),
            ((batch * 37, 4, 64), torch.bfloat16),
            ((batch * 37, 4), torch.float32),
            ((batch * 37, 4, 4), torch.float32),
        )
    )

    def region(*values):
        return mhc_combine(*values, coefficient_precision=precision)

    with deterministic_kernels(True):
        pair = capture_graph_pair(
            make_fx(region)(*inputs),
            inputs,
            original_output=region(*inputs),
            memory_budget=budget,
        )
        axis = ChunkAxis(0, before=batch)
        config = AxisChunks(
            11, {i: axis for i in range(4)}, rules=operator_chunk_rules(batch)
        )
        forward = chunk_artifact(pair.forward, (axis,) * 4, 37, config)
        with torch.no_grad():
            expected = pair.forward.graph_module(*inputs)
            actual = executable(forward)(*inputs)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            saved = forward.chunking.outputs[-pair.saved_value_count :]
            backward = chunk_artifact(pair.backward, (*saved, axis), 37, config)
            tangent = torch.randn_like(expected[0])
            reference = pair.backward.graph_module(
                *expected[-pair.saved_value_count :], tangent
            )
            gradients = executable(backward)(
                *actual[-pair.saved_value_count :], tangent
            )
            torch.testing.assert_close(gradients, reference, rtol=0, atol=0)


def test_text_policy_slices_head_major_attention_before_residual_projection():
    attention = torch.randn(2, 17, 3, 5, dtype=torch.float64).transpose(1, 2)
    residual = torch.randn(2, 17, 15, dtype=torch.float64)
    weight = torch.randn(15, 15, dtype=torch.float64)
    inputs = (attention, residual, weight)

    def region(attention, residual, weight):
        return (residual + attention.transpose(1, 2).view(2, 17, 15) @ weight,)

    graph = make_fx(region)(*inputs)
    config = TextAxisChunks(17, 2, 7).configure_stage(
        StageInfo(
            "projection",
            (),
            tuple(TensorGeometry.from_tensor(x) for x in inputs),
            (
                TaskInputRole.ACTIVATION,
                TaskInputRole.ACTIVATION,
                TaskInputRole.PARAMETER,
            ),
            tuple(str(n.target) for n in graph.graph.nodes if n.op == "call_function"),
            False,
        )
    )
    assert config is not None
    assert dict(config.inputs) == {0: ChunkAxis(2), 1: ChunkAxis(1)}
    full = GraphArtifact.capture(
        kind="forward", graph_module=graph, example_inputs=inputs
    )
    chunked = chunk_artifact(full, (ChunkAxis(2), ChunkAxis(1), None), 17, config)
    torch.testing.assert_close(executable(chunked)(*inputs), region(*inputs))


@pytest.mark.cuda
@pytest.mark.parametrize("budget", [0.0, 1.0])
def test_linear_attention_head_groups_preserve_complete_packed_recurrence(budget):
    if not torch.cuda.is_available():
        pytest.skip("requires device provider kernels")
    import importlib

    from mlops.dispatch import deterministic_kernels

    provider = importlib.import_module("mlops.providers.fla.hybrid")
    generator = torch.Generator(device="cuda").manual_seed(731)

    def draw(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, device="cuda", generator=generator, dtype=dtype)

    with torch.no_grad():
        values = (
            torch.nn.functional.normalize(draw(259, 4, 32), dim=-1),
            torch.nn.functional.normalize(draw(259, 4, 32), dim=-1),
            draw(259, 8, 32),
            draw(259, 8).sigmoid(),
            draw(259, 8),
            draw(8, dtype=torch.float32) * 0.1,
            draw(8, dtype=torch.float32) * 0.1,
        )
    inputs = (
        *(value.requires_grad_() for value in values),
        torch.tensor([0, 128, 259], dtype=torch.int64, device="cuda"),
        torch.tensor(
            [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]],
            dtype=torch.int64,
            device="cuda",
        ),
    )
    variant = "save" if budget else "recompute"
    config = TextAxisChunks(
        259, 1, 7, (variant,), linear_attention_head_chunk_size=3
    ).configure_stage(
        StageInfo(
            "recurrence",
            (),
            tuple(TensorGeometry.from_tensor(value) for value in inputs),
            (TaskInputRole.ACTIVATION,) * 5
            + (TaskInputRole.PARAMETER,) * 2
            + (TaskInputRole.CONTROL,) * 2,
            ("mlops.linear_attention_fla_gated_delta_rule_fwd.default",),
            False,
        )
    )
    assert config is not None

    def region(*values):
        return provider.linear_apply(*values, scale=32**-0.5)

    with deterministic_kernels(True):
        pair = capture_graph_pair(
            make_fx(region)(*inputs),
            inputs,
            original_output=region(*inputs),
            memory_budget=budget,
        )
        forward = chunk_artifact(
            pair.forward,
            tuple(config.inputs.get(i) for i in range(len(inputs))),
            4,
            config,
        )
        with torch.no_grad():
            expected = pair.forward.graph_module(*inputs)
            actual = executable(forward)(*inputs)
            saved = forward.chunking.outputs[-pair.saved_value_count :]
            backward = chunk_artifact(
                pair.backward, (*saved, ChunkAxis(1, after=2)), 4, config
            )
            tangent = torch.randn_like(expected[0])
            full = pair.backward.graph_module(
                *expected[-pair.saved_value_count :], tangent
            )
            gradients = executable(backward)(
                *actual[-pair.saved_value_count :], tangent
            )
        for value, reference in zip(
            (actual[0], *gradients), (expected[0], *full), strict=True
        ):
            if reference is None:
                assert value is None
                continue
            relative = (
                value.float() - reference.float()
            ).norm() / reference.float().norm().clamp_min(1e-8)
            assert relative < 0.02, float(relative)


@pytest.mark.cuda
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("mode", ["topk_then_softmax", "sigmoid_noaux_tc"])
@pytest.mark.parametrize("distribution", ["random", "single_pair"])
def test_routed_rows_restore_order_and_accumulate_statistics_and_gradients(
    batch, mode, distribution
):
    if not torch.cuda.is_available():
        pytest.skip("requires device provider kernels")
    from mlops.dispatch import deterministic_kernels
    from mlops.lora.experts import expert_lora

    generator = torch.Generator(device="cuda").manual_seed(731)
    rows, width, experts, hidden, rank = 37, 64, 8, 64, 8

    def draw(*shape, dtype=torch.bfloat16, train=True):
        value = torch.randn(*shape, generator=generator, device="cuda")
        return (value * shape[-1] ** -0.5).to(dtype).requires_grad_(train)

    inputs = (
        draw(batch, rows, width),
        draw(batch, rows, width),
        draw(experts, width, dtype=torch.float32),
        draw(experts, 2 * hidden, width, train=False),
        draw(experts, width, hidden, train=False),
        draw(experts, rank, width, dtype=torch.float32),
        draw(experts, 2 * hidden, rank, dtype=torch.float32),
        draw(experts, rank, hidden, dtype=torch.float32),
        draw(experts, width, rank, dtype=torch.float32),
    )
    if distribution == "single_pair":
        with torch.no_grad():
            inputs[0].abs_().add_(0.25)
            inputs[2].fill_(-0.05)
            inputs[2][0].fill_(0.05)
            inputs[2][1].fill_(0.025)

    def region(*values):
        output, _auxiliary, counts, probability_sum = expert_lora(
            *values,
            top_k=2,
            routing_mode=mode,
            scale=0.5,
            implementation="builtin",
        )
        # Nonlinear balancing consumes these complete statistics in its own
        # stage. Local sort indices and offsets never cross this boundary.
        return output, counts, probability_sum

    with deterministic_kernels(True):
        pair = capture_graph_pair(
            make_fx(region)(*inputs),
            inputs,
            original_output=region(*inputs),
            memory_budget=0.0,
        )
        config = AxisChunks(11, {0: 1, 1: 1}, rules=operator_chunk_rules(batch))
        forward = chunk_artifact(
            pair.forward, (ChunkAxis(1), ChunkAxis(1), *(None,) * 7), rows, config
        )
        with torch.no_grad():
            expected = pair.forward.graph_module(*inputs)
            actual = executable(forward)(*inputs)
            torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
            if distribution == "single_pair":
                assert torch.count_nonzero(actual[1][..., 2:]) == 0
            torch.testing.assert_close(actual[2], expected[2], rtol=2e-6, atol=2e-6)
            saved = forward.chunking.outputs[-pair.saved_value_count :]
            backward = chunk_artifact(
                pair.backward, (*saved, ChunkAxis(1), None), rows, config
            )
            tangents = torch.randn_like(expected[0]), torch.randn_like(expected[2])
            full = pair.backward.graph_module(
                *expected[-pair.saved_value_count :], *tangents
            )
            gradients = executable(backward)(
                *actual[-pair.saved_value_count :], *tangents
            )
            if distribution == "single_pair":
                for gradient in gradients[5:]:
                    assert torch.count_nonzero(gradient[2:]) == 0
        for value, reference in zip(
            (actual[0], *gradients), (expected[0], *full), strict=True
        ):
            if reference is None:
                assert value is None
                continue
            assert torch.isfinite(value).all()
            relative = (
                value.float() - reference.float()
            ).norm() / reference.float().norm().clamp_min(1e-8)
            assert relative < 0.02, float(relative)
