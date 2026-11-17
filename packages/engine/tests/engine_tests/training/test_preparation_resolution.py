"""CPU preset/case expansion, exact overrides and scheduled-padding qualification."""

import subprocess
import sys
from dataclasses import replace

import pytest

from engine_tests.fixtures.definition_operations import definitions
from engine_tests.fixtures.external_definition import SOURCE
from shadowspill.parallelism import (
    GroupRef,
    InputFieldPlacement,
    InputPlacement,
    Parallelism,
)
from shadowspill.planning.memory_search.algorithm import SearchOptions
from shadowspill.training.definitions import (
    ModelDefinition,
    PreparationContext,
    PreparationDefinition,
    Registry,
)
from shadowspill.training.definitions.preparation import ResolvedPreparationDefinition
from shadowspill.training.definitions.stacked import STACKED_PRESET
from shadowspill.training.preparation.configuration.expansion import expand_schedules
from shadowspill.training.preparation.configuration.resolution import (
    resolve_preparation,
)
from shadowspill_client import (
    BatchConfig,
    BuildCandidates,
    BuildCase,
    BuildOptions,
    PlanningOptions,
    Policy,
    Preparation,
)
from shadowspill_client.inputs.coverage_bounds import CoverageError
from shadowspill_client.inputs.shapes import BufferShape
from shadowspill_client.inputs.validation import combined_schema
from shadowspill_client.protocol.codec import decode, encode
from shadowspill_client.protocol.inputs import Capacity, InputSchema, TensorField


def stacked_context():
    model, objective, _ = definitions()
    schema = combined_schema(model.inputs, objective.inputs)
    placement = InputPlacement.independent(
        1, fields=[InputFieldPlacement(tuple(f.path)) for f in schema.fields]
    )
    return PreparationContext(
        model,
        schema,
        (BatchConfig(limits={"max_examples": 4}),),
        Parallelism(inputs=placement),
    )


def token_context(placement=None):
    registry = Registry()
    definition = ModelDefinition(
        "external/sequence@1",
        dict,
        lambda _: pytest.fail("constructed during description"),
        InputSchema(
            [TensorField(("x",), "float32", ("positions",), "pack", 0)],
            [
                Capacity("tokens", "positions", "sum", ("x",), 0, scheduled=True),
                Capacity("length", "positions", "max", ("x",), 0),
                Capacity("records", "sequences", "count"),
            ],
        ),
    )
    registry.add_model(definition, source=SOURCE)
    model = registry.resolve_model(definition.id)
    schema = combined_schema(model.inputs, InputSchema([]))
    placement = placement or InputPlacement.independent(
        1, fields=[InputFieldPlacement(("inputs", "x"))]
    )
    world = sum(len(g) for g in placement.partitions.groups)
    return PreparationContext(
        model,
        schema,
        (BatchConfig(limits={"tokens": 128, "length": 64, "records": 3}),),
        Parallelism(ranks_per_replica=world, inputs=placement),
    )


def token_buffers(context, batch, paths, case):
    capacity = case.microbatch_size * int(context.batches[batch].limits["length"])
    return [
        BufferShape(3, {path: (capacity,) for path in paths})
    ] * context.parallelism.inputs.partition_count


def preset(settings, *, buffers=token_buffers):
    return ResolvedPreparationDefinition(
        PreparationDefinition("external/preparation@1", lambda _: settings, buffers),
        SOURCE,
    )


def test_builtin_inspection_is_fully_resolved_pinned_and_has_constructive_coverage():
    context = stacked_context()
    registry = Registry()
    installed = registry.resolve_preparation(STACKED_PRESET)
    resolved = resolve_preparation(installed, context)
    assert decode(encode(resolved)) == resolved
    assert [c.microbatch_size for c in resolved.cases] == [1, 2, 4]
    assert resolved.planning.microbatch_counts == (1, 2)
    assert resolved.planning.workers == 0
    assert resolved.origins["planning.workers"] == "engine_default"
    assert resolved.planning.to_dict()["search"] == SearchOptions().to_dict()
    paths = [tuple(f.path) for f in context.schema.fields]
    expanded = expand_schedules(resolved, installed, context, paths)
    assert len(expanded.candidates) == 5 and len(expanded.rejected) == 1
    assert expanded.build_cases == 3
    assert expanded.coverage(context)[0].max_datums == 4
    with pytest.raises(ValueError, match="source changed"):
        registry.resolve_preparation(STACKED_PRESET, source=SOURCE)
    with pytest.raises(TypeError):
        resolved.origins["new"] = "explicit"


def test_explicit_lists_replace_presets_and_fixed_none_keeps_unchunked():
    context = token_context()
    chunk = Policy("external/chunk@1", {"size": 32})
    graph = Policy("external/graph@1")
    installed = preset(
        Preparation(
            candidates=BuildCandidates(
                microbatch_sizes=[1, 2],
                chunking=[None, chunk],
                graph_pair_policies=[None, graph],
            ),
            planning=PlanningOptions(microbatch_counts=[1, 2, 4], workers=7),
        )
    )
    assert len(resolve_preparation(installed, context).cases) == 8
    changed = resolve_preparation(
        installed,
        context,
        Preparation(
            build=BuildOptions(chunking=None),
            candidates=BuildCandidates(
                microbatch_sizes=[2], graph_pair_policies=[None]
            ),
            planning=PlanningOptions(microbatch_counts=[1]),
        ),
    )
    assert len(changed.cases) == 1
    assert changed.cases[0].build.values["chunking"] is None
    assert changed.planning.microbatch_counts == (1,)
    assert changed.planning.workers == 7
    assert changed.origins["candidates.microbatch_sizes"] == "explicit"
    assert changed.origins["planning.workers"] == "preset"
    assert (
        resolve_preparation(
            installed, context, Preparation(planning=PlanningOptions(workers=0))
        ).planning.workers
        == 0
    )


@pytest.mark.parametrize(
    "inherited,explicit,expected,origin",
    [
        (None, None, 512 << 20, "engine_default"),
        (768 << 20, None, 768 << 20, "preset"),
        (768 << 20, 0, 0, "explicit"),
        (None, 1024 << 20, 1024 << 20, "explicit"),
    ],
)
def test_worker_headroom_resolves_without_changing_task_timing_policy(
    inherited, explicit, expected, origin
):
    installed = preset(
        Preparation(
            candidates=BuildCandidates(microbatch_sizes=[1]),
            external_headroom_bytes=inherited,
        )
    )
    resolved = resolve_preparation(
        installed, token_context(), Preparation(external_headroom_bytes=explicit)
    )
    assert decode(encode(resolved)) == resolved
    assert resolved.external_headroom_bytes == expected
    assert resolved.origins["external_headroom_bytes"] == origin
    assert resolved.origins["profiling"] == "engine_default"


def test_builtin_preserves_coverage_for_each_explicit_batch_option():
    context = stacked_context()
    context = replace(
        context, batches=(BatchConfig(limits={"max_examples": 3}), *context.batches)
    )
    installed = Registry().resolve_preparation(STACKED_PRESET)
    resolved = resolve_preparation(installed, context)
    assert [case.microbatch_size for case in resolved.cases] == [1, 2, 3, 4]
    paths = [tuple(field.path) for field in context.schema.fields]
    proofs = expand_schedules(resolved, installed, context, paths).coverage(context)
    assert [proof.max_datums for proof in proofs] == [3, 4]


def test_complete_cases_keep_coupling_and_track_case_overrides():
    context = token_context()
    chunk = Policy("external/chunk@1")
    installed = preset(
        Preparation(
            build=BuildOptions(chunking=chunk),
            candidates=BuildCandidates(microbatch_sizes=[1, 2]),
        )
    )
    requested = Preparation(
        candidates=BuildCandidates(
            cases=[
                BuildCase(1, BuildOptions(chunking=None)),
                BuildCase(2),
            ]
        )
    )
    resolved = resolve_preparation(installed, context, requested)
    assert len(resolved.cases) == 2
    assert resolved.cases[0].build.values["chunking"] is None
    assert resolved.cases[1].build.values["chunking"] == chunk.to_dict()
    assert resolved.origins["cases.0.chunking"] == "explicit"
    assert resolved.origins["cases.1.chunking"] == "preset"
    coupled = preset(requested)
    with pytest.raises(ValueError, match="complete case overrides"):
        resolve_preparation(
            coupled,
            context,
            Preparation(candidates=BuildCandidates(microbatch_sizes=[2])),
        )


def test_q25_filters_before_build_and_proves_only_requested_larger_shape():
    context = token_context()
    installed = preset(
        Preparation(
            candidates=BuildCandidates(microbatch_sizes=[1, 2]),
            planning=PlanningOptions(microbatch_counts=[1, 2, 3]),
        )
    )
    paths = [("inputs", "x")]
    resolved = resolve_preparation(installed, context)
    expanded = expand_schedules(resolved, installed, context, paths)
    assert [
        (resolved.cases[c.case].microbatch_size, c.count) for c in expanded.candidates
    ] == [(1, 1), (1, 2), (2, 1)]
    assert len(expanded.rejected) == 3
    assert expanded.rejected[0].capacities == {"tokens": 192}
    proof = expanded.coverage(context)[0]
    assert proof.schedule.buffers[0].fields[paths[0]] == (128,)
    smaller = resolve_preparation(
        installed,
        context,
        Preparation(candidates=BuildCandidates(microbatch_sizes=[1])),
    )
    with pytest.raises(CoverageError):
        expand_schedules(smaller, installed, context, paths).coverage(context)
    assert [case.microbatch_size for case in smaller.cases] == [1]


@pytest.mark.parametrize("cooperating", [True, False])
def test_global_capacity_counts_partitions_once_not_cooperating_workers(cooperating):
    placement = InputPlacement(
        GroupRef("inputs", ((0, 1),) if cooperating else ((0,), (1,))),
        fields=[InputFieldPlacement(("inputs", "x"))],
    )
    context = token_context(placement)
    installed = preset(
        Preparation(
            candidates=BuildCandidates(microbatch_sizes=[1]),
            planning=PlanningOptions(microbatch_counts=[1, 2]),
        )
    )
    resolved = resolve_preparation(installed, context)
    expanded = expand_schedules(resolved, installed, context, [("inputs", "x")])
    assert len(expanded.candidates) == (2 if cooperating else 1)
    assert expanded.candidates[-1].schedule.capacities(context.schema)["tokens"] == 128


def test_expansion_bounds_precede_callbacks_and_huge_counts_precede_allocation():
    context = token_context()
    installed = preset(Preparation(candidates=BuildCandidates(microbatch_sizes=[1, 2])))
    with pytest.raises(ValueError, match="build-case expansion"):
        resolve_preparation(installed, context, max_build_cases=1)
    resolved = resolve_preparation(
        installed,
        context,
        Preparation(planning=PlanningOptions(microbatch_counts=[10**20])),
    )
    expanded = expand_schedules(
        resolved, installed, context, [("inputs", "x")], max_buffers=1
    )
    assert not expanded.candidates and len(expanded.rejected) == 2

    def broken(*args):
        pytest.fail("geometry evaluated despite case limit")

    installed = replace(
        installed, definition=replace(installed.definition, buffers=broken)
    )
    with pytest.raises(ValueError, match="schedule-case expansion"):
        expand_schedules(
            resolved, installed, context, [("inputs", "x")], max_schedule_cases=1
        )


def test_unknown_search_keys_and_unexpected_geometry_errors_are_not_silently_dropped():
    context = token_context()
    installed = preset(Preparation(candidates=BuildCandidates(microbatch_sizes=[1])))
    search = SearchOptions().to_dict()
    search["misspelled"] = 1
    with pytest.raises(ValueError, match="complete core SearchOptions"):
        resolve_preparation(
            installed, context, Preparation(planning=PlanningOptions(search=search))
        )

    def broken(*args):
        raise RuntimeError("definition defect")

    resolved = resolve_preparation(installed, context)
    installed = replace(
        installed, definition=replace(installed.definition, buffers=broken)
    )
    with pytest.raises(RuntimeError, match="definition defect"):
        expand_schedules(resolved, installed, context, [("inputs", "x")])


def test_preset_registry_rechecks_source_and_description_has_no_framework_import():
    registry = Registry()
    installed = preset(Preparation(candidates=BuildCandidates(microbatch_sizes=[1])))
    registry.add_preparation(installed.definition, source=SOURCE)
    assert registry.resolve_preparation(installed.definition.id) == installed
    with pytest.raises(ValueError, match="already registered"):
        registry.add_preparation(installed.definition, source=SOURCE)
    script = """
import sys
for name in ('torch', 'triton', 'shadowspill_workloads'):
    sys.modules[name] = None
from engine_tests.training.test_preparation_resolution import stacked_context
from shadowspill.training.definitions import Registry
from shadowspill.training.preparation.configuration.resolution import (
    resolve_preparation,
)
from shadowspill.training.preparation.configuration.expansion import expand_schedules
context = stacked_context()
definition = Registry().resolve_preparation('shadowspill/stacked@1')
resolved = resolve_preparation(definition, context)
paths = [tuple(f.path) for f in context.schema.fields]
assert expand_schedules(resolved, definition, context, paths).coverage(context)
"""
    subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True
    )
