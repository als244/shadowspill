"""Qualify explicit SonicMoE BF16 LoRA axis contracts before model integration."""

import json
from collections import Counter
from math import ceil

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._python_dispatch import TorchDispatchMode

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.aot import capture_graph_pair
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.chunking import (
    AxisChunks,
    AxisFrom,
    ChunkAxis,
    ChunkLocal,
    OperatorRule,
)
from shadowspill.pytorch.chunking.geometry import LocalLayout
from shadowspill.pytorch.chunking.transform import chunk_artifact
from tests.shadowspill.pytorch.chunking.test_axis_vjp import executable
from workloads.models.shared.chunking import operator_rules


def sonic_rules(batch):
    """Candidate contracts; no readonly, replay, or FP8 certification here."""
    local = ChunkLocal(0)
    specifications = {
        "sonicmoe_routing_metadata": ((None, local, local, local), (0,)),
        "lora_sonicmoe_up_fwd": ((local, local, local), ()),
        "lora_sonicmoe_up_bwd": ((AxisFrom(1), None, None, None), (1, 2, 3)),
        "lora_sonicmoe_down_fwd": ((AxisFrom(10), local), ()),
        "lora_sonicmoe_down_bwd": (
            (local, None, None, None, AxisFrom(9)),
            (1, 2, 3),
        ),
    }
    return (
        *operator_rules(batch),
        *(
            OperatorRule(
                f"mlops.{name}.default", "bf16-lora-candidate-1", axes, reductions
            )
            for name, (axes, reductions) in specifications.items()
        ),
    )


def inputs_for(batch, distribution, *, device="cpu"):
    sequence, rank = (37, 8) if batch == 1 else (65, 3)
    generator = torch.Generator().manual_seed(731)

    def draw(shape, dtype=torch.bfloat16, train=True):
        value = torch.randn(shape, generator=generator) * shape[-1] ** -0.5
        return value.to(device=device, dtype=dtype).requires_grad_(train)

    inputs = (
        draw((batch, sequence, 64)),
        draw((batch, sequence, 64)),
        draw((8, 64), torch.float32),
        draw((8, 80, 64), train=False),
        draw((8, 64, 40), train=False),
        draw((8, rank, 64), torch.float32),
        draw((8, 80, rank), torch.float32),
        draw((8, rank, 40), torch.float32),
        draw((8, 64, rank), torch.float32),
    )
    if distribution == "single_pair":
        with torch.no_grad():
            inputs[0].abs_().add_(0.25)
            inputs[2].fill_(-0.05)
            inputs[2][0].fill_(0.05)
            inputs[2][1].fill_(0.025)
    return inputs


def region_for(mode, implementation="sonicmoe"):
    if implementation == "sonicmoe":
        # Calling the provider directly permits CPU fake capture without
        # claiming that a real CPU tensor can execute a CUDA-only kernel.
        from mlops.providers.sonicmoe.lora import expert_lora
    else:
        from mlops.lora.experts import expert_lora

    def region(*values):
        output, _, counts, probabilities = expert_lora(
            *values,
            top_k=2,
            routing_mode=mode,
            scale=0.5,
            **({"implementation": "builtin"} if implementation == "builtin" else {}),
        )
        return output, counts, probabilities

    return region


def captured(inputs, mode, batch):
    region = region_for(mode)
    with FakeTensorMode(allow_non_fake_inputs=True) as fake:
        fake_inputs = tuple(fake.from_tensor(value) for value in inputs)
        graph = make_fx(region)(*fake_inputs)
        pair = capture_graph_pair(
            graph, fake_inputs, original_output=graph(*fake_inputs), memory_budget=0.0
        )
        size = 11 if batch == 1 else 17
        axis = ChunkAxis(1)
        config = AxisChunks(size, {0: axis, 1: axis}, rules=sonic_rules(batch))
        forward = chunk_artifact(
            pair.forward, (axis, axis, *(None,) * 7), inputs[0].shape[1], config
        )
        saved = forward.chunking.outputs[-pair.saved_value_count :]
        backward = chunk_artifact(
            pair.backward, (*saved, axis, None), inputs[0].shape[1], config
        )
    return pair, forward, backward


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("mode", ["topk_then_softmax", "sigmoid_noaux_tc"])
def test_sonic_contract_capture_and_local_escape_refusal(batch, mode):
    values = inputs_for(batch, "random")
    pair, forward, backward = captured(values, mode, batch)
    for artifact in (forward, backward):
        assert artifact.chunking.rows == values[0].shape[1]
        sizes = {
            size
            for size, _ in (
                *artifact.chunking.children,
                *artifact.chunking.accumulating_children,
            )
        }
        assert sizes == {
            artifact.chunking.size,
            artifact.chunking.rows % artifact.chunking.size,
        }
        assert all(
            not isinstance(value, LocalLayout) for value in artifact.chunking.outputs
        )
    assert forward.storage_contract == pair.forward.storage_contract

    from mlops.providers.sonicmoe.moe import routing_metadata

    ids = torch.zeros((batch * 37, 2), dtype=torch.int32)
    graph = make_fx(lambda x: routing_metadata(x, 8), tracing_mode="fake")(ids)
    artifact = GraphArtifact.capture(
        kind="forward", graph_module=graph, example_inputs=(ids,)
    )
    with pytest.raises(CaptureError, match="local"):
        chunk_artifact(
            artifact,
            (ChunkAxis(0, before=batch),),
            37,
            AxisChunks(11, {0: ChunkAxis(0, before=batch)}, rules=sonic_rules(batch)),
        )
    assert not torch.cuda.is_initialized()


class Calls(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.counts = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if "sonicmoe" in str(func):
            self.counts[str(func)] += 1
        return func(*args, **(kwargs or {}))


def close(actual, expected, *, bound, name):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.isfinite(actual).all()
    relative = (
        actual.float() - expected.float()
    ).norm() / expected.float().norm().clamp_min(1e-8)
    assert relative < bound, (name, float(relative), bound)
    return float(relative)


@pytest.mark.cuda
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("mode", ["topk_then_softmax", "sigmoid_noaux_tc"])
@pytest.mark.parametrize("distribution", ["random", "single_pair"])
def test_sonic_chunks_preserve_routing_and_arbitrary_vjp(batch, mode, distribution):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("requires H100 SonicMoE provider qualification")
    from mlops.dispatch.context import weight_gradients_at
    from mlops.providers.sonicmoe import lora

    inputs = inputs_for(batch, distribution, device="cuda")
    support = lora.supports(inputs[0], inputs[3], inputs[4])
    assert support, support.reason
    with weight_gradients_at(torch.float32):
        pair, forward, backward = captured(inputs, mode, batch)
        ordinary = region_for(mode)(*inputs)
        torch.manual_seed(918)
        tangents = torch.randn_like(ordinary[0]), torch.randn_like(ordinary[2])
        eager = torch.autograd.grad(
            (ordinary[0], ordinary[2]),
            [v for v in inputs if v.requires_grad],
            tangents,
        )
        builtin = region_for(mode, "builtin")(*inputs)
        independent = torch.autograd.grad(
            (builtin[0], builtin[2]),
            [v for v in inputs if v.requires_grad],
            tangents,
        )
        with torch.no_grad(), Calls() as calls:
            actual = executable(forward)(*inputs)
            forward_calls = calls.counts.copy()
            calls.counts.clear()
            gradients = executable(backward)(
                *actual[-pair.saved_value_count :], *tangents
            )
            backward_calls = calls.counts.copy()
    chunks = ceil(inputs[0].shape[1] / forward.chunking.size)
    for name in ("lora_sonicmoe_up_fwd", "lora_sonicmoe_down_fwd"):
        assert forward_calls[f"mlops.{name}.default"] == chunks
        assert backward_calls[f"mlops.{name}.default"] == chunks
    for name in ("lora_sonicmoe_up_bwd", "lora_sonicmoe_down_bwd"):
        assert backward_calls[f"mlops.{name}.default"] == chunks
    torch.testing.assert_close(actual[1], ordinary[1], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], builtin[1], rtol=0, atol=0)
    torch.testing.assert_close(actual[2], ordinary[2], rtol=2e-6, atol=2e-6)
    if distribution == "single_pair":
        assert actual[1][2:].count_nonzero() == 0
        assert all(g[2:].count_nonzero() == 0 for g in gradients[5:])
    values = (actual[0], *(g for g in gradients if g is not None))
    names = (
        "output",
        "hidden",
        "residual",
        "router",
        "up_a",
        "up_b",
        "down_a",
        "down_b",
    )
    comparisons = {}
    for name, references in (
        ("ordinary_sonic", (ordinary[0], *eager)),
        ("independent_builtin", (builtin[0], *independent)),
    ):
        comparisons[name] = {
            label: close(
                value,
                reference,
                bound=(
                    0.02
                    if name == "ordinary_sonic"
                    else 0.03
                    if value.dtype == torch.bfloat16
                    else 0.01
                ),
                name=label,
            )
            for label, value, reference in zip(names, values, references, strict=True)
        }
    print(
        json.dumps(
            dict(
                batch=batch,
                sequence=inputs[0].shape[1],
                rank=inputs[5].shape[1],
                mode=mode,
                distribution=distribution,
                chunks=chunks,
                forward_calls=forward_calls,
                backward_calls=backward_calls,
                relative_errors=comparisons,
            )
        ),
        flush=True,
    )
