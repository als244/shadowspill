"""Source-pinned portable policies preserve existing packed-text choices."""

import subprocess
import sys
from dataclasses import replace

import pytest

from shadowspill.training.components import load_registry
from shadowspill.training.preparation.capture.templates import capture_inventory
from shadowspill.training.preparation.configuration.layouts import prepare_layouts
from shadowspill.training.preparation.configuration.policies import (
    materialize_build_options,
)
from shadowspill.training.preparation.setup import resolve_training
from shadowspill_client import (
    BatchConfig,
    BuildCandidates,
    BuildCase,
    BuildOptions,
    Memory,
    Objective,
    PlanningOptions,
    Policy,
    Preparation,
)
from shadowspill_workloads.components import source


def preparation():
    return Preparation(
        candidates=BuildCandidates(
            cases=[
                BuildCase(
                    size,
                    BuildOptions(
                        chunking=Policy(
                            "shadowspill/text_chunks@1",
                            {"token_capacity": size, "chunk_size": size // 2},
                        ),
                        graph_pair_policy=Policy(
                            "shadowspill/text_retention@1",
                            {"name": "cost-attention", "chunked": True},
                        ),
                    ),
                )
                for size in (8, 16)
            ]
        ),
        planning=PlanningOptions(microbatch_counts=[1, 2]),
    )


def test_readme_dp_candidates_keep_full_updates_and_independent_moe_geometry():
    from shadowspill.parallelism import Parallelism
    from shadowspill_client import Model, Optimizer, Weights
    from shadowspill_client.parallelism import data_parallel
    from shadowspill_client.text import build_candidates

    registry = load_registry(["shadowspill-workloads"])
    setup = resolve_training(
        registry,
        model=Model(
            "qwen3moe", config={"preset": "30b-a3b"}, initialize=Weights.random(seed=0)
        ),
        objective=Objective.sum(
            {"tokens": Objective("cross_entropy"), "balance": Objective("moe_balance")},
            weights={"balance": 0.001},
        ),
        optimizer=Optimizer("adamw"),
        parallelism=Parallelism.from_dict(data_parallel(8, min_shard_bytes=0)),
        batch=BatchConfig(
            limits={
                "token_capacity": 2097152,
                "max_sequence_length": 4096,
                "max_sequences": 8192,
            }
        ),
        memory=Memory(20, 96),
        prepare=Preparation(
            candidates=build_candidates(
                microbatch_tokens=[8192, 16384, 32768],
                projection_chunks=[0, 8192],
            )
        ),
    )
    prepared = setup.config.preparation
    assert prepared.planning.microbatch_counts == (1, 8, 16, 32)
    assert len(prepared.cases) == 6
    assert resolve_training(registry, setup.config).config == setup.config
    for case in prepared.cases:
        chunks = materialize_build_options(
            registry, case.build, setup.config.policy_sources
        )["chunking"]
        if chunks is not None:
            assert chunks.chunking.chunk_size == 8192
            assert chunks.chunking.expert_chunk_size == case.microbatch_size
    inventory = capture_inventory(prepare_layouts(setup))
    full_sizes = set()
    for candidates in inventory.families.values():
        for candidate in candidates:
            total = 0
            for use in candidate.workers.values():
                template = inventory.templates[use.main.template]
                (size,) = template.binding.inputs.layout.buffer.fields[
                    "inputs", "tokens"
                ]
                total += size * use.main.count
            assert total <= 2097152
            if total == 2097152:
                full_sizes.add(size)
    assert full_sizes == {8192, 16384, 32768}


def test_explicit_token_candidates_derive_counts_but_explicit_counts_win():
    from workload_tests.components.test_moe_components import settings

    options = dict(
        model=settings("qwen3moe"),
        objective=Objective("cross_entropy"),
        operations={"forward_backward": ["metrics"]},
        batch=BatchConfig(
            limits={"token_capacity": 64, "max_sequence_length": 4, "max_sequences": 16}
        ),
        memory=Memory(1, 1),
    )
    registry = load_registry(["shadowspill-workloads"])
    candidates = BuildCandidates(microbatch_sizes=[8, 32])
    inferred = resolve_training(
        registry, **options, prepare=Preparation(candidates=candidates)
    )
    explicit = resolve_training(
        registry,
        **options,
        prepare=Preparation(
            candidates=candidates, planning=PlanningOptions(microbatch_counts=[1, 2])
        ),
    )
    assert inferred.config.preparation.planning.microbatch_counts == (1, 2, 8)
    assert explicit.config.preparation.planning.microbatch_counts == (1, 2)


def test_complete_packed_cases_keep_geometry_policy_sources_and_scheduled_cap():
    from workload_tests.components.test_moe_components import settings

    registry = load_registry(["shadowspill-workloads"])
    setup = resolve_training(
        registry,
        model=settings("qwen3moe"),
        objective=Objective("shadowspill/cross_entropy@1"),
        operations={"forward_backward": ["metrics"], "forward": ["logprobs"]},
        batch=BatchConfig(
            limits={"token_capacity": 16, "max_sequence_length": 4, "max_sequences": 4}
        ),
        memory=Memory(1, 1),
        prepare=preparation(),
    )
    assert resolve_training(registry, setup.config).config == setup.config
    inventory = capture_inventory(prepare_layouts(setup))
    capacities = set()
    for template in inventory.templates.values():
        values = materialize_build_options(
            registry, template.build, setup.config.policy_sources
        )
        chunks = values["chunking"]
        (capacity,) = template.binding.inputs.layout.buffer.fields["inputs", "tokens"]
        assert chunks.chunking.sequence_length == capacity
        assert chunks.chunking.batch_size == 1
        assert chunks.chunking.chunk_size == chunks.chunking.expert_chunk_size
        assert chunks.chunking.chunk_size == capacity // 2
        assert chunks.variants == ("save", "recompute")
        assert chunks.lifetimes is not None
        capacities.add(capacity)
    assert capacities == {8, 16}
    for candidates in inventory.families.values():
        for candidate in candidates:
            for use in candidate.workers.values():
                template = inventory.templates[use.main.template]
                (capacity,) = template.binding.inputs.layout.buffer.fields[
                    "inputs", "tokens"
                ]
                assert use.main.count * capacity <= 16


@pytest.mark.parametrize("name", ["cost-input", "cost-attention", "attention-input"])
@pytest.mark.parametrize("chunked", [False, True])
def test_retention_component_keeps_existing_recipe_choices(name, chunked):
    from shadowspill_workloads.policies.text import build_options

    registry = load_registry(["shadowspill-workloads"])
    policy = registry.resolve_policy(
        "shadowspill/text_retention@1",
        kind="graph_pair_policy",
        config={"name": name, "chunked": chunked},
    )
    expected = build_options(
        {
            "graph_pair_policy": name,
            "chunk_size": 4 if chunked else 0,
            "sequence_length": 8,
        },
        1,
    )["graph_pair_policy"]
    assert policy.create().identity() == expected.identity()


@pytest.mark.parametrize(
    "config",
    [
        {"token_capacity": 0, "chunk_size": 4},
        {"token_capacity": 8, "chunk_size": True},
        {"token_capacity": 8, "chunk_size": 4.0},
        {"token_capacity": 8, "chunk_size": 4, "expert_chunk_size": 0},
        {"token_capacity": 2**31, "chunk_size": 4},
    ],
)
def test_invalid_chunk_geometry_fails_before_policy_construction(config):
    registry = load_registry(["shadowspill-workloads"])
    with pytest.raises(ValueError, match="text chunk geometry"):
        registry.resolve_policy(
            "shadowspill/text_chunks@1", kind="chunking", config=config
        )


def test_policy_provider_change_invalidates_saved_source(monkeypatch):
    registry = load_registry(["shadowspill-workloads"])
    policy = registry.resolve_policy(
        "shadowspill/text_retention@1", kind="graph_pair_policy"
    )
    monkeypatch.setattr(source, "_provider_digest", lambda name: b"changed provider")
    changed = load_registry(["shadowspill-workloads"])
    with pytest.raises(ValueError, match="source changed"):
        changed.resolve_policy(
            policy.component.id,
            kind="graph_pair_policy",
            config=policy.config,
            source=policy.source,
        )
    with pytest.raises(ValueError, match="source changed"):
        registry.resolve_policy(
            policy.component.id,
            kind="graph_pair_policy",
            source=replace(policy.source, digest="0" * 64),
        )


def test_policy_description_and_resolution_import_no_repository_or_framework():
    script = """
import sys
from shadowspill.training.components import load_registry
from workload_tests.components.test_text_policy_components import preparation
registry = load_registry(['shadowspill-workloads'])
for case in preparation().candidates.cases:
    for name, value in case.build.to_dict().items():
        resolved = registry.resolve_policy(
            value['name'], kind=name, config=value['config'])
        assert resolved.config
assert not any(name in sys.modules for name in (
    'torch', 'triton', 'mlops', 'fla', 'quack', 'sonicmoe', 'cutlass'))
assert not any(name.split('.')[0] in {'benchmarks', 'reference', 'qualification'}
               for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], text=True, capture_output=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr
