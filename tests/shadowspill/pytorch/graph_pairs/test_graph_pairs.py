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
    partition_training_capture,
    saved_value_footprint,
)
from shadowspill.pytorch.partition import PartitionedExport, partition_export
from shadowspill.pytorch.partition.differentiability import (
    differentiable_output_positions,
)


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


class _IndexedEmbedding(nn.Module):
    """Positions computed at this module's own scope, then two children.

    GPT-2 is shaped this way. `torch.arange` runs in the body's forward,
    before any child, so the automatic policy leaves it out of every anchored
    block and makes it a prologue of its own -- a stage whose only output is
    an index.
    """

    def __init__(self) -> None:
        super().__init__()
        self.values = nn.Embedding(8, 4)
        self.positions = nn.Embedding(8, 4)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        order = torch.arange(tokens.shape[-1], device=tokens.device)
        return self.values(tokens) + self.positions(order)


class _ControlPrologueNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = _IndexedEmbedding()
        self.head = nn.Linear(4, 4)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(tokens))


def _capture_control_prologue() -> tuple[FakeTensorMode, object, tuple[object, ...]]:
    """Capture the objective, with real values for the authentic index."""

    model = _ControlPrologueNetwork()
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    replica = fake_device_model(model, mode)
    tokens = torch.tensor([1, 5, 2, 7], dtype=torch.int64)
    target = torch.randn(4, 4)
    inputs = fake_device_inputs([tokens, target], mode)

    def objective(
        current: nn.Module, value: torch.Tensor, expected: torch.Tensor
    ) -> torch.Tensor:
        return torch.nn.functional.mse_loss(current(value), expected)

    with mode:
        captured = capture_training(replica, objective, inputs)
    # Deriving the index runs its producer slice for real, so the roots have
    # to be real tensors of the captured geometry, as planning supplies.
    roots = tuple(
        torch.zeros(tuple(value.shape), dtype=value.dtype)
        if isinstance(value, torch.Tensor)
        else value
        for value in captured.exported.flat_inputs
    )
    return mode, captured, roots


def test_a_control_only_prologue_is_folded_into_the_stage_that_uses_it() -> None:
    """A stage with only an index to show for itself is not a training stage.

    It holds no activation, produces no gradient and receives no cotangent,
    so a boundary there costs a task and saves nothing. Left standing it
    refuses the whole model, which is what GPT-2 did.
    """

    mode, captured, roots = _capture_control_prologue()
    with mode:
        standing = partition_export(
            captured.exported,
            captured.capture_module,
            representative_root_inputs=roots,
        )
        folded = partition_training_capture(
            captured, representative_root_inputs=roots
        ).partitioned

    assert not differentiable_output_positions(standing.stages[0].output)
    assert len(folded.stages) == len(standing.stages) - 1
    assert all(differentiable_output_positions(stage.output) for stage in folded.stages)
    with mode:
        assert len(capture_training_stages(folded)) == len(folded.stages)


def _capture(
    dtype: torch.dtype = torch.float32,
) -> tuple[FakeTensorMode, PartitionedExport]:
    model = _RepeatedNetwork().to(dtype)
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    replica = fake_device_model(model, mode)
    inputs = fake_device_inputs(
        [torch.randn(2, 8, dtype=dtype), torch.randn(2, 8, dtype=dtype)], mode
    )

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


def test_rounding_an_accumulation_once_is_a_form_of_its_own() -> None:
    """At bf16, adding a multiply's gradient inside the multiply rounds the
    sum once where adding after rounds it twice, so the store derives the
    accumulating form each way it is asked for, and only the one asked for
    adds inside the multiply."""

    mode, partitioned = _capture(torch.bfloat16)
    store = GraphPairStore()
    adding_inside = torch.ops.shadowspill.accumulate_matmul_.default

    def added_inside(round_once: bool) -> set[bool]:
        with mode:
            stages = capture_training_stages(
                partitioned,
                graph_pair_store=store,
                accumulating=True,
                round_accumulation_once=round_once,
            )
        return {
            adding_inside
            in {node.target for node in item.pair.backward.graph_module.graph.nodes}
            for stage in stages
            for item in stage.graph_pairs.options(accumulates=True)
        }

    assert added_inside(False) == {False}
    assert added_inside(True) == {True}
    assert added_inside(False) == {False}


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
