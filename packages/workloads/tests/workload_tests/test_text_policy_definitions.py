"""Source-pinned portable policies preserve existing packed-text choices."""

import subprocess
import sys
from dataclasses import replace

import pytest

from shadowspill.training.definitions import load_registry
from shadowspill.training.preparation.layouts import prepare_layouts
from shadowspill.training.preparation.policies import materialize_build_options
from shadowspill.training.preparation.setup import resolve_training
from shadowspill.training.preparation.templates import capture_inventory
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
from shadowspill_workloads.definitions import source


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


def test_complete_packed_cases_keep_geometry_policy_sources_and_scheduled_cap():
    from workload_tests.test_moe_definitions import settings

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
def test_retention_definition_keeps_existing_recipe_choices(name, chunked):
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
            policy.definition.id,
            kind="graph_pair_policy",
            config=policy.config,
            source=policy.source,
        )
    with pytest.raises(ValueError, match="source changed"):
        registry.resolve_policy(
            policy.definition.id,
            kind="graph_pair_policy",
            source=replace(policy.source, digest="0" * 64),
        )


def test_policy_description_and_resolution_import_no_recipe_or_framework():
    script = """
import sys
from shadowspill.training.definitions import load_registry
from workload_tests.test_text_policy_definitions import preparation
registry = load_registry(['shadowspill-workloads'])
for case in preparation().candidates.cases:
    for name, value in case.build.to_dict().items():
        resolved = registry.resolve_policy(
            value['definition'], kind=name, config=value['config'])
        assert resolved.config
assert not any(name in sys.modules for name in (
    'torch', 'triton', 'mlops', 'fla', 'quack', 'sonicmoe', 'cutlass'))
assert not any(name.startswith('shadowspill_workloads.recipes.')
               for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], text=True, capture_output=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr
