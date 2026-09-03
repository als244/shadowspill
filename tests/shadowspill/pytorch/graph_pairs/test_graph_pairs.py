from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode

from shadowspill.pytorch.capture import aot as aot_module
from shadowspill.pytorch.capture.aot import capture_training
from shadowspill.pytorch.capture.fake import fake_device_inputs, fake_device_model
from shadowspill.pytorch.graph_pairs import (
    DifferentiatedStage,
    GraphPairStore,
    capture_training_stages,
    saved_value_footprint,
)
from shadowspill.pytorch.partition import PartitionedExport, partition_export


class _RepeatedNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block() for _ in range(4)])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            value = torch.relu(block(value))
        return value


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(8, 8) for _ in range(2)])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        result = self.experts[0](value) + self.experts[1](value)
        assert isinstance(result, torch.Tensor)
        return result


def _capture() -> tuple[FakeTensorMode, PartitionedExport]:
    model = _RepeatedNetwork()
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    replica = fake_device_model(model, mode)
    inputs = fake_device_inputs([torch.randn(2, 8), torch.randn(2, 8)], mode)

    def objective(
        current: nn.Module, value: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return torch.nn.functional.mse_loss(current(value), target)

    with mode:
        captured = capture_training(replica, objective, inputs)
        partitioned = partition_export(captured.exported, captured.capture_module)
    return mode, partitioned


def test_each_training_stage_has_endpoint_graph_pairs() -> None:
    mode, partitioned = _capture()
    with mode:
        stages = capture_training_stages(partitioned)

    assert partitioned.repeated_groups == ("model.blocks",)
    assert len(stages) == 4
    assert all(
        tuple(
            (item.option_id, item.memory_budget) for item in stage.graph_pairs.variants
        )
        == (("save", None), ("recompute", 0.0))
        for stage in stages
    )
    save_footprints = tuple(
        saved_value_footprint(stage.graph_pairs.variant("save").pair)
        for stage in stages
    )
    recompute_footprints = tuple(
        saved_value_footprint(stage.graph_pairs.variant("recompute").pair)
        for stage in stages
    )
    assert all(item.internal_minimum_bytes == 0 for item in recompute_footprints)
    assert sum(item.internal_minimum_bytes for item in save_footprints) > 0
    assert sum(item.internal_minimum_bytes for item in save_footprints) > sum(
        item.internal_minimum_bytes for item in recompute_footprints
    )
    assert all(
        str(stage.graph_pairs.variant("save").pair.forward.graph_module.graph)
        != str(stage.graph_pairs.variant("recompute").pair.forward.graph_module.graph)
        for stage in stages
    )
    assert all(
        stage.graph_pairs.variant("save").pair.backward.operator_targets
        for stage in stages
    )
    assert all(
        stage.graph_pairs.variant("recompute").pair.forward.operator_targets
        for stage in stages
    )
    assert all(
        stage.graph_pairs.variant("save").pair.specialized_unit_tangent_count == 0
        for stage in stages[:-1]
    )
    assert (
        stages[-1].graph_pairs.variant("save").pair.specialized_unit_tangent_count == 1
    )
    assert (
        stages[-1].graph_pairs.variant("recompute").pair.specialized_unit_tangent_count
        == 1
    )


def test_the_accumulating_form_is_derived_only_when_asked_for() -> None:
    """Deriving the second form costs a capture, so nobody pays for it unasked.

    A step whose microbatches never accumulate never asks, and the contract
    keeps only the form it will run. Asking once grows the structural entry,
    so every later occurrence rebinds rather than deriving again.
    """

    mode, partitioned = _capture()
    store = GraphPairStore()
    with mode:
        plain = capture_training_stages(partitioned, graph_pair_store=store)
        accumulating = capture_training_stages(
            partitioned, graph_pair_store=store, accumulating=True
        )

    assert all(
        not any(item.accumulates for item in stage.graph_pairs.variants)
        for stage in plain
    ), "a capture that never accumulates carries only the captured form"

    pairs = accumulating[0].graph_pairs
    derived = pairs.options(accumulates=True)
    captured = pairs.options(accumulates=False)
    assert len(derived) == len(captured)
    assert all(item.accumulates for item in derived)
    assert tuple(item.option_id for item in derived) == tuple(
        item.option_id for item in captured
    )
    assert all(
        len(new.pair.backward.example_arguments)
        > len(old.pair.backward.example_arguments)
        for old, new in zip(captured, derived, strict=True)
    )


def test_recompute_budget_is_bound_to_lazy_partition_callback() -> None:
    mode, partitioned = _capture()
    with (
        aot_module.functorch_config.patch(activation_memory_budget=1.0),
        mode,
    ):
        stages = capture_training_stages(partitioned)

    assert all(
        saved_value_footprint(
            stage.graph_pairs.variant("recompute").pair
        ).internal_minimum_bytes
        == 0
        for stage in stages
    )


def test_repeated_stage_occurrences_share_one_structural_inventory() -> None:
    mode, partitioned = _capture()
    repository = GraphPairStore()
    with mode:
        stages = capture_training_stages(
            partitioned,
            graph_pair_store=repository,
        )

    assert repository.misses == 3
    assert repository.hits == 1
    first_interior = stages[1].graph_pairs.variant("save").pair
    second_interior = stages[2].graph_pairs.variant("save").pair
    assert first_interior.forward.graph_module is second_interior.forward.graph_module
    assert first_interior.backward.graph_module is second_interior.backward.graph_module
    assert first_interior.backward is not second_interior.backward
    assert (
        first_interior.forward.compatibility_digest
        == second_interior.forward.compatibility_digest
    )
    first_storages = tuple(
        value.untyped_storage()._cdata
        for value in first_interior.forward.example_arguments
        if isinstance(value, torch.Tensor)
    )
    second_storages = tuple(
        value.untyped_storage()._cdata
        for value in second_interior.forward.example_arguments
        if isinstance(value, torch.Tensor)
    )
    assert first_storages != second_storages


def test_graph_pair_store_persists_structural_inventories(tmp_path: Path) -> None:
    mode, partitioned = _capture()
    with mode:
        first = GraphPairStore(tmp_path)
        expected = capture_training_stages(
            partitioned,
            graph_pair_store=first,
        )
        second = GraphPairStore(tmp_path)
        actual = capture_training_stages(
            partitioned,
            graph_pair_store=second,
        )

    assert first.unique_keys == 3
    assert first.misses == 3
    assert second.unique_keys == 3
    assert second.misses == 0
    assert second.hits == 4
    assert tuple(tmp_path.rglob("graph_pairs.pt"))
    assert tuple(_digests(stage) for stage in actual) == tuple(
        _digests(stage) for stage in expected
    )
    for expected_stage, actual_stage in zip(expected, actual, strict=True):
        for expected_variant, actual_variant in zip(
            expected_stage.graph_pairs.variants,
            actual_stage.graph_pairs.variants,
            strict=True,
        ):
            for expected_artifact, actual_artifact in (
                (expected_variant.pair.forward, actual_variant.pair.forward),
                (expected_variant.pair.backward, actual_variant.pair.backward),
            ):
                assert str(actual_artifact.graph_module.graph) == str(
                    expected_artifact.graph_module.graph
                )


def _digests(stage: DifferentiatedStage) -> tuple[tuple[str, str], ...]:
    return tuple(
        (
            item.pair.forward.compatibility_digest,
            item.pair.backward.compatibility_digest,
        )
        for item in stage.graph_pairs.variants
    )
