"""Default text shapes stay separate from logical capacity and execution policy."""

from types import SimpleNamespace

import pytest

from shadowspill.training.definitions import load_registry
from shadowspill.training.preparation.family_checks import _orderings
from shadowspill.training.preparation.layouts import prepare_layouts
from shadowspill.training.preparation.setup import resolve_training
from shadowspill.training.preparation.templates import capture_inventory
from shadowspill_client import BatchConfig, Memory, Objective, Optimizer
from shadowspill_workloads.inputs.text.preparation import defaults

from .test_llama_definition import settings


@pytest.mark.parametrize(
    "cap,length,partitions,sizes,counts",
    [
        (131072, 8192, 1, (8192, 16384, 32768), (1, 4, 8, 16)),
        (2097152, 4096, 8, (4096, 8192, 16384), (1, 16, 32, 64)),
        (12, 4, 1, (4, 8, 12), (1, 3)),
        (8, 16, 1, (8,), (1,)),
        (8, 4, 2, (4,), (1,)),
    ],
)
def test_default_shapes_follow_sequence_bound_and_logical_input_partitions(
    cap, length, partitions, sizes, counts
):
    context = SimpleNamespace(
        model=SimpleNamespace(config={"architecture": {"max_seq_len": length}}),
        batches=[
            BatchConfig(
                limits={
                    "token_capacity": cap,
                    "max_sequence_length": length,
                    "max_sequences": 1024,
                }
            )
        ],
        parallelism=SimpleNamespace(inputs=SimpleNamespace(partition_count=partitions)),
    )
    result = defaults(context, preset="fixture")
    assert result.candidates.microbatch_sizes == sizes
    assert result.planning.microbatch_counts == counts
    assert not result.build.values
    assert result.candidates.chunking is result.candidates.graph_pair_policies is None


def test_resolved_default_compares_full_steps_and_every_depth_breadth_ordering():
    registry = load_registry(["shadowspill", "shadowspill-workloads"])
    setup = resolve_training(
        registry,
        model=settings(capacity=8192),
        objective=Objective("cross_entropy"),
        optimizer=Optimizer("adamw"),
        batch=BatchConfig(
            limits={
                "token_capacity": 131072,
                "max_sequence_length": 8192,
                "max_sequences": 1024,
            }
        ),
        memory=Memory(24, 128),
    )
    cases = setup.config.preparation.cases
    assert [case.microbatch_size for case in cases] == [8192, 16384, 32768]
    assert all(case.build.values["chunking"] is None for case in cases)
    assert all(case.build.values["graph_pair_policy"] is None for case in cases)
    expanded = next(iter(setup.expansions.values()))
    assert {
        (cases[row.case].microbatch_size, row.count)
        for row in expanded.candidates
        if cases[row.case].microbatch_size * row.count == 131072
    } == {(8192, 16), (16384, 8), (32768, 4)}
    assert all(
        cases[row.case].microbatch_size * row.count <= 131072
        for row in expanded.candidates
    )
    inventory = capture_inventory(prepare_layouts(setup))
    # Counts and traversals reuse three shape captures plus their masked forms.
    assert len(inventory.templates) == 6
    for family in inventory.families.values():
        for row in family:
            count = row.layout.candidate.count
            assert [(order.depth, order.breadth) for order in _orderings(row)] == [
                (depth, count // depth)
                for depth in range(1, count + 1)
                if count % depth == 0
            ]
