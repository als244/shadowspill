"""Default text shapes stay separate from logical capacity and execution policy."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from shadowspill.training.definitions import load_registry
from shadowspill.training.preparation.capture.templates import capture_inventory
from shadowspill.training.preparation.configuration.layouts import prepare_layouts
from shadowspill.training.preparation.planning.validation import _orderings
from shadowspill.training.preparation.setup import resolve_training
from shadowspill_client import BatchConfig, Memory, Objective, Optimizer, Preparation
from shadowspill_client.text import build_candidates
from shadowspill_workloads.inputs.text.preparation import defaults
from workload_tests.definitions.test_llama_definition import settings


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
        requested=Preparation(),
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
    registry = load_registry(["shadowspill-workloads"])
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
    # Counts and traversals reuse the three requested training captures.
    assert len(inventory.templates) == 3
    for family in inventory.families.values():
        for row in family:
            count = row.layout.candidate.count
            assert [(order.depth, order.breadth) for order in _orderings(row)] == [
                (depth, count // depth)
                for depth in range(1, count + 1)
                if count % depth == 0
            ]


@pytest.mark.parametrize("maximum,expected", [(None, 128), (16, 16), (256, 256)])
def test_sequence_capacity_is_per_microbatch_and_kernel_bound_is_per_sequence(
    maximum, expected
):
    setup = resolve_training(
        load_registry(["shadowspill-workloads"]),
        model=settings(capacity=131072),
        objective=Objective("cross_entropy"),
        optimizer=Optimizer("adamw"),
        batch=BatchConfig(
            limits={
                "token_capacity": 2097152,
                "max_sequence_length": 4096,
                "max_sequences": 8192,
            }
        ),
        memory=Memory(24, 128),
        prepare=Preparation(
            candidates=build_candidates(
                microbatch_tokens=[65536],
                max_sequences_per_microbatch=maximum,
            )
        ),
    )
    inventory = capture_inventory(prepare_layouts(setup))
    for template in inventory.templates.values():
        inputs = template.binding.inputs
        assert inputs.layout.buffer.max_datums == expected
        assert inputs.layout.limits["max_sequence_length"] == 4096
        assert template.model_build.config["sequence_length_bound"] == 4096
        assert template.model_build.config["architecture"]["max_seq_len"] == 131072
        assert inputs.shapes["inputs", "sequence_lengths"] == (expected + 16,)


def test_distinct_sequence_bounds_do_not_reuse_the_same_capture():
    batch = BatchConfig(
        limits={
            "token_capacity": 16,
            "max_sequence_length": 4,
            "max_sequences": 8,
        }
    )
    setup = resolve_training(
        load_registry(["shadowspill-workloads"]),
        model=settings(capacity=16),
        objective=Objective("cross_entropy"),
        optimizer=Optimizer("adamw"),
        memory=Memory(1, 1),
        batch=[
            batch,
            replace(batch, limits={**batch.limits, "max_sequence_length": 8}),
        ],
        prepare=Preparation(candidates=build_candidates(microbatch_tokens=[16])),
    )
    inventory = capture_inventory(prepare_layouts(setup))
    assert {
        t.model_build.config["sequence_length_bound"]
        for t in inventory.templates.values()
    } == {4, 8}
