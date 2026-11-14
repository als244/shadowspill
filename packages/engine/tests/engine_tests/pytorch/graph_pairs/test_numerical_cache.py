"""Differentiation caches honor settings used by an opaque autograd formula."""

import json

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from shadowspill.pytorch.capture.artifacts import TaskInputProvenance
from shadowspill.pytorch.graph_pairs import GraphPairStore
from shadowspill.pytorch.numerics import capture_numerical_settings
from shadowspill.pytorch.partition.artifacts import (
    Stage,
    StageExample,
    StageValueSource,
)
from shadowspill.task.inputs import TaskInputRole


@torch.library.custom_op("numerical_cache_test::seven", mutates_args=())
def seven(x: torch.Tensor) -> torch.Tensor:
    return x * 7


@seven.register_fake
def seven_fake(x):
    return torch.empty_like(x)


def backward(ctx, gradient):
    # Two mathematically equivalent algorithms, selected when AOT captures
    # backward. The forward graph alone cannot key this choice.
    if torch.are_deterministic_algorithms_enabled():
        return gradient * 7
    return (
        ((((gradient + gradient) + gradient) + gradient) + gradient) + gradient
    ) + gradient


seven.register_autograd(backward)


@pytest.mark.parametrize("persistent", [False, True])
def test_numeric_configuration_specializes_differentiation_and_disk_cache(
    tmp_path, persistent
):
    x = torch.randn(7, requires_grad=True)
    graph = make_fx(lambda x: (seven(x),))(x)
    stage = Stage(
        "test",
        "test",
        graph,
        (StageValueSource(root_input_index=0),),
        (TaskInputProvenance(TaskInputRole.USER_INPUT),),
        (),
        (0,),
    )
    example = StageExample(stage, (x,), graph(x))
    store = GraphPairStore(tmp_path if persistent else None)
    enabled = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()

    def resolve(store):
        return store.resolve(
            example, (0,), specialize_unit_tangents=False, variants=("save",)
        ).reference.backward.graph_module

    try:
        torch.use_deterministic_algorithms(False)
        first = resolve(store)
        first_settings = capture_numerical_settings()
        torch.use_deterministic_algorithms(True)
        second = resolve(store)
        second_settings = capture_numerical_settings()
        assert store.unique_keys == store.misses == 2 and store.hits == 0
        fresh = resolve(GraphPairStore())
        assert first.code != second.code == fresh.code
        assert any(
            node.target == torch.ops.aten.mul.Tensor for node in second.graph.nodes
        )
        if persistent:
            reader = GraphPairStore(tmp_path)
            assert resolve(reader).code == second.code
            assert reader.hits == 1 and reader.misses == 0
            assert {
                json.loads(path.read_text())["numerical_settings"]
                for path in tmp_path.rglob("manifest.json")
            } == {first_settings, second_settings}
        torch.use_deterministic_algorithms(False)
        assert resolve(store).code == first.code
        assert store.hits == 1 and store.misses == 2
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn)
