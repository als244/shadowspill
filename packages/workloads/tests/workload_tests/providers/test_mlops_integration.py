"""Optional engine integration: actual capture, destination paths and refusals."""

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

pytest.importorskip("mlops")

from mlops.providers.sonicmoe import lora

from shadowspill.pytorch import AxisFrom, ChunkLocal
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.capture.gradients import (
    accumulate_gradient_outputs,
)
from shadowspill_workloads.integrations.mlops import (
    mhc_chunk_rules,
    operator_preparations,
    sonicmoe_chunk_rules,
    sonicmoe_gradient_destination_rules,
    sonicmoe_preparation,
)


def arguments(direction, dtype=torch.float32, *, train_base=False):
    experts, tokens, top_k, width, inner, rank = 4, 5, 2, 16, 8, 8
    rows = tokens * top_k

    def tensor(*shape):
        return torch.empty(shape, dtype=torch.bfloat16)

    offsets = torch.empty(experts + 1, dtype=torch.int32)
    gather = torch.empty(rows, dtype=torch.int32)
    scatter = torch.empty(rows, dtype=torch.int32)
    reverse = torch.empty(rows, dtype=torch.int32)
    a = tensor(experts, rank, width if direction == "up" else inner)
    b = tensor(experts, 2 * inner if direction == "up" else width, rank)
    weight = tensor(
        experts, *([2 * inner, width] if direction == "up" else [width, inner])
    )
    weight_shape = list(weight.shape) if train_base else [0, 0, 0]
    z = tensor(rows, rank)
    if direction == "up":
        args = (
            tensor(rows, 2 * inner),
            tensor(tokens, width),
            gather,
            reverse,
            weight,
            a,
            b,
            z,
            offsets,
            weight_shape,
            list(a.shape),
            list(b.shape),
            top_k,
            dtype,
            [],
        )
    else:
        args = (
            tensor(tokens, width),
            tensor(rows, 2 * inner),
            weight,
            a,
            b,
            z,
            offsets,
            gather,
            scatter,
            torch.empty(rows),
            weight_shape,
            list(a.shape),
            list(b.shape),
            dtype,
            None,
            [],
        )
    destinations = [torch.empty_like(a, dtype=dtype), torch.empty_like(b, dtype=dtype)]
    return args, destinations


def capture(direction, dtype, transform="canonical"):
    args, _destinations = arguments(direction, dtype)
    operation = getattr(lora, f"_{direction}_backward")
    positions = [i for i, value in enumerate(args) if isinstance(value, torch.Tensor)]
    tensors = tuple(args[i] for i in positions)

    def function(*values):
        supplied = list(args)
        for index, value in zip(positions, values, strict=True):
            supplied[index] = value
        result = operation(*supplied)
        a, b = result[2:4]
        if transform == "canonical" and direction == "up":
            experts, packed, rank = b.shape
            b = b.view(experts, packed // 2, 2, rank).transpose(1, 2).reshape_as(b)
        elif transform == "cast":
            b = b.to(torch.bfloat16).to(dtype)
        elif transform == "crop":
            b = b[:, :, :-1]
        elif transform == "two_scales":
            b = b * 2.0
        elif transform not in ("identity", "canonical"):
            raise AssertionError(transform)
        return a, b * 1.75

    graph = make_fx(function, tracing_mode="fake")(*tensors)
    return GraphArtifact.capture(
        kind="backward", graph_module=graph, example_inputs=tensors
    )


@pytest.mark.parametrize("direction", ["up", "down"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("permission", [False, True])
def test_actual_operator_groups_destinations_and_respects_narrow_permission(
    direction, dtype, permission
):
    initialized = torch.cuda.is_initialized()
    original = capture(direction, dtype)
    code = original.graph_module.code
    rules = sonicmoe_gradient_destination_rules()
    transformed = accumulate_gradient_outputs(
        original,
        (0, 1),
        gradient_destination_rules=rules,
        fuse_narrow_gradient_accumulation=permission,
    )
    calls = [
        node
        for node in transformed.graph_module.graph.nodes
        if str(node.target) == rules[direction == "down"].replacement
    ]
    eligible = dtype == torch.float32 or permission
    assert len(calls) == int(eligible)
    if eligible:
        node = calls[0]
        named = {
            argument.name: value
            for argument, value in zip(
                node.target._schema.arguments, node.args, strict=False
            )
        }
        named.update(node.kwargs)
        assert named["output_indices"] == [2, 3]
        assert named["gradient_scales"] == [1.0, 1.75]
        assert named["canonical_b"] == (direction == "up")
        assert named["fuse_narrow_gradient_accumulation"] == permission
        assert named["contract_version"] == 1
        assert len(named["destinations"]) == 2
    assert original.graph_module.code == code
    assert torch.cuda.is_initialized() == initialized


@pytest.mark.parametrize("direction", ["up", "down"])
@pytest.mark.parametrize("transform", ["cast", "crop", "two_scales"])
def test_unsupported_b_path_preserves_a_fusion_and_b_fallback(direction, transform):
    original = capture(direction, torch.float32, transform)
    rules = sonicmoe_gradient_destination_rules()
    transformed = accumulate_gradient_outputs(
        original,
        (0, 1),
        gradient_destination_rules=rules,
        fuse_narrow_gradient_accumulation=True,
    )
    calls = [
        node
        for node in transformed.graph_module.graph.nodes
        if str(node.target) == rules[direction == "down"].replacement
    ]
    assert len(calls) == 1
    node = calls[0]
    index = next(
        i
        for i, argument in enumerate(node.target._schema.arguments)
        if argument.name == "output_indices"
    )
    assert node.args[index] == [2]
    # Its unconsumed B still has the original shape; it is neither silently
    # dropped nor rounded at the destination instead of at the explicit cast.
    assert node.meta["val"][3].numel() > 0
    assert "aten.add_.Tensor" in transformed.operator_targets


def test_chunk_rules_keep_sorted_rows_local_and_match_mutating_outputs():
    rules = {rule.target: rule for rule in sonicmoe_chunk_rules()}
    assert len(rules) == 7
    for direction in ("up", "down"):
        fresh = rules[f"mlops.lora_sonicmoe_{direction}_bwd.default"]
        accumulating = rules[f"mlops.lora_sonicmoe_{direction}_bwd_accumulate.default"]
        assert fresh.output_axes == accumulating.output_axes
        assert fresh.reductions == accumulating.reductions == (1, 2, 3)
    metadata = rules["mlops.sonicmoe_routing_metadata.default"]
    assert metadata.output_axes == (None, ChunkLocal(0), ChunkLocal(0), ChunkLocal(0))
    assert metadata.reductions == (0,)
    assert rules["mlops.lora_sonicmoe_up_bwd.default"].output_axes[0] == AxisFrom(1)
    fp8 = sonicmoe_chunk_rules(compute_precision="fp8_block")
    assert len(fp8) == 5
    for rule in fp8:
        assert rule.output_axes == rules[rule.target].output_axes
        assert rule.reductions == rules[rule.target].reductions
        assert rule.version != rules[rule.target].version
        assert "accumulate" not in rule.target
    with pytest.raises(ValueError, match="frozen fp8_block"):
        sonicmoe_chunk_rules(compute_precision="fp8_current")


def test_preparation_descriptions_are_explicit_and_need_no_device(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("description queried a device")

    monkeypatch.setattr(torch.cuda, "get_device_capability", forbidden)
    explicit = sonicmoe_preparation(cache_root="/tmp/build-only")
    assert explicit.factory == "mlops.preparation.sonic:factory"
    assert dict(explicit.options) == {"cache_root": "/tmp/build-only"}
    ordinary = operator_preparations()
    assert len(ordinary) == 2
    assert ordinary[0].factory == "mlops.preparation.available:factory"
    assert (
        ordinary[1].factory
        == "mlops.providers.builtin.sparse_latent_preparation:factory"
    )
    assert operator_preparations(dtype=torch.float32) == ()
    assert len(mhc_chunk_rules()) == 4
    assert len(sonicmoe_chunk_rules(compute_precision="fp8_block")) == 5
