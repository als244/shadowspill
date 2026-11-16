"""Saved capability inventory is strict; native execution is qualified separately."""

import subprocess
import sys
import zipfile
from dataclasses import replace

import pytest

from engine_tests.offline._example import communication, options, profiled
from engine_tests.training.test_allocation_coverage import certify, compiler_fixture
from engine_tests.training.test_preparation_templates import inventory
from engine_tests.training.test_setup_resolution import installed
from shadowspill import MemoryBudgets, plan_program
from shadowspill.artifacts._values import plain
from shadowspill.parallelism import InputFieldPlacement, InputPlacement, Parallelism
from shadowspill.planning.shared import SharedMemory
from shadowspill.step.ordering import StepDataOrdering
from shadowspill.training.definitions import ModelReferenceDefinition
from shadowspill.training.preparation import bundle_validation, family, family_selection
from shadowspill.training.preparation.bundle import TrainingBundle
from shadowspill.training.preparation.bundle_manifest import manifest
from shadowspill.training.preparation.family_identity import (
    candidate_identity,
    execution_names,
)
from shadowspill.training.preparation.family_records import (
    PlannedCandidate,
    PreparedPlans,
)
from shadowspill.training.preparation.family_selection import certificate, joint_memory
from shadowspill_client import (
    BatchConfig,
    BuildCandidates,
    PlanningOptions,
    Preparation,
    Weights,
)
from shadowspill_client.inputs.coverage_bounds import CoverageError


@pytest.fixture
def saved_family(monkeypatch):
    return make_bundle(monkeypatch)


def make_bundle(monkeypatch, **settings):
    # This static CPU fixture admits a 100-byte device and has no CUDA overhead.
    prepare = replace(settings.get("prepare", Preparation()), external_headroom_bytes=0)
    settings = {**settings, "prepare": prepare}
    prepared = inventory(**settings)
    base = profiled()
    profiling = prepared.layouts.setup.config.preparation.profiling.to_dict()
    sources = {
        key: replace(
            base,
            program=certify(template, compiler_fixture(template)),
            task_profiles={
                key: replace(p, metadata={"memory_kind": "static"})
                for key, p in base.task_profiles.items()
            },
            metadata={
                "memory_kind": "static",
                "preparation_template": key,
                "profiling_scenario": template.signature()["scenario"],
                "profiling_options": profiling,
                "external_headroom_bytes": (
                    prepared.layouts.setup.config.preparation.external_headroom_bytes
                ),
            },
        )
        for key, template in prepared.templates.items()
    }
    native = plan_program(
        base,
        memory_budgets=MemoryBudgets(100, 64),
        communication_info=communication(),
        search_options=options(),
    )
    checked = []
    # Only native schedule/placement validation is substituted. Source pins,
    # operation dependencies, layouts and constructive coverage stay real.
    monkeypatch.setattr(
        bundle_validation, "validate_plan", lambda *args: checked.append(args[1])
    )
    monkeypatch.setattr(
        family_selection,
        "shared_memory",
        lambda primary, additional: SharedMemory({}, {}, 0, 80, 64),
    )

    def no_search(*args, **kwargs):
        pytest.fail("saved bundle invoked planning search")

    monkeypatch.setattr(family, "plan_program", no_search)
    rows, proofs = {}, {}
    plans = {
        rank: {}
        for rank in range(prepared.layouts.setup.context.parallelism.world_size)
    }
    for key, candidates in prepared.families.items():
        for source in candidates:
            if (key, source.layout.candidate.batch) in proofs:
                continue
            ordering = (
                None
                if key.operation == "forward"
                else StepDataOrdering.depth_first(source.workers[0].main.count)
            )
            worker = execution_names(source, ordering)
            candidate = PlannedCandidate(
                candidate_identity(source, ordering),
                source,
                ordering,
                dict.fromkeys(source.workers, worker),
                native.timeline.makespan_ns / 1e9,
            )
            try:
                proof = certificate(prepared, candidate)
            except CoverageError:
                continue
            for rank in source.workers:
                for name in worker.members():
                    plans[rank][name] = native
            rows[key] = (*rows.get(key, ()), candidate)
            proofs[key, candidate.batch] = proof
    selected, memory = joint_memory([c for rows in rows.values() for c in rows], plans)
    result = PreparedPlans(prepared, rows, selected, proofs, memory, (), 1)
    bundle = TrainingBundle.from_prepared(result, sources)
    checked.clear()
    return bundle, checked


def test_bundle_reconstructs_shared_names_and_rejects_rank_local_renaming(monkeypatch):
    fields = inventory().layouts.setup.context.schema.fields
    bundle, checked = make_bundle(
        monkeypatch,
        parallelism=Parallelism.dp(
            num_replicas=2,
            inputs=InputPlacement.independent(
                2, fields=[InputFieldPlacement(tuple(f.path)) for f in fields]
            ),
        ),
    )
    restored = bundle.recheck(installed())
    assert restored.plans[0].keys() == restored.plans[1].keys()
    assert len(checked) == sum(map(len, bundle.plans.values()))
    metadata = plain(bundle.metadata)
    metadata["candidates"][0]["workers"][1]["main"] = "renamed-local-plan"
    with pytest.raises(ValueError, match="Plan identities"):
        replace(bundle, metadata=metadata).recheck(installed())
    first = next(iter(bundle.plans[1]))
    renamed = dict(bundle.plans[1])
    renamed["d" * 64] = renamed.pop(first)
    with pytest.raises(ValueError, match="required worker Plan"):
        replace(bundle, plans={0: bundle.plans[0], 1: renamed}).recheck(installed())


def test_bundle_keeps_every_explicit_batch_option(monkeypatch):
    bundle, _ = make_bundle(
        monkeypatch,
        batch=[BatchConfig(limits={"max_examples": n}) for n in (3, 4)],
        prepare=Preparation(
            candidates=BuildCandidates(microbatch_sizes=[2, 3, 4]),
            planning=PlanningOptions(microbatch_counts=[1]),
        ),
    )
    restored = bundle.recheck(installed())
    assert len(restored.coverage) == 6
    metadata = plain(bundle.metadata)
    metadata["candidates"].pop()
    with pytest.raises(ValueError, match="every requested operation/batch"):
        replace(bundle, metadata=metadata).recheck(installed())


def test_bundle_roundtrip_preserves_all_dependencies_and_deduplicates_code(
    saved_family, tmp_path
):
    bundle, checked = saved_family
    path = tmp_path / "training.ssp"
    bundle.save(path)
    restored = TrainingBundle.load(path)
    actual = restored.recheck(installed())
    assert restored.to_dict() == bundle.to_dict()
    assert manifest(actual) == plain(bundle.metadata)
    assert len(checked) == len(bundle.plans[0])
    assert len(restored.sources) < len(actual.inventory.templates)
    with zipfile.ZipFile(path) as archive:
        assert len([n for n in archive.namelist() if n.startswith("blobs/")]) == 1
    with pytest.raises(TypeError):
        restored.plans[0]["extra"] = next(iter(restored.plans[0].values()))
    assert "weights" not in restored.to_dict()


@pytest.mark.parametrize("kind", ["random", "pretrained", "artifact", "checkpoint"])
def test_saved_bundle_reuses_computation_with_different_initial_weights(
    saved_family, tmp_path, kind
):
    bundle, _ = saved_family
    registry = installed()
    weights = {
        "random": Weights.random(seed=123),
        "pretrained": replace(
            Weights.pretrained("fixture/model", revision="a" * 40),
            metadata={"files": [{"name": "weights", "checksum": "abc"}]},
        ),
        "artifact": Weights.artifact("uploaded-weights"),
        "checkpoint": None,
    }[kind]
    sources = dict(bundle.config.definition_sources)
    reference = None
    if kind == "pretrained":
        reference = "external/reference@1"
        sources["reference"] = sources["model"]
        registry.add_model_reference(
            ModelReferenceDefinition(reference, lambda _: True, lambda model: model),
            source=sources["reference"],
        )
    config = replace(
        bundle.config,
        model=replace(bundle.config.model, initialize=weights),
        reference_definition=reference,
        definition_sources=sources,
        origins={**bundle.config.origins, "model.config.num_classes": "explicit"},
    )
    rebound = bundle.with_initialization(config)
    assert rebound.config == config and bundle.config != config
    assert rebound.sources == bundle.sources and rebound.plans == bundle.plans
    path = tmp_path / "rebound.ssp"
    rebound.save(path)
    restored = TrainingBundle.load(path)
    assert manifest(restored.recheck(registry)) == plain(bundle.metadata)
    # A pretrained bundle must also permit a new random source without retaining
    # resolver fields or altering any captured computation.
    assert rebound.with_initialization(bundle.config).config == bundle.config


@pytest.mark.parametrize(
    "change",
    [
        lambda c: replace(c, model=replace(c.model, config={"num_classes": 4})),
        lambda c: replace(
            c, objective=replace(c.objective, defaults={"temperature": 2.0})
        ),
        lambda c: replace(c, optimizer=replace(c.optimizer, defaults={"lr": 0.2})),
        lambda c: replace(c, memory=replace(c.memory, device_gib=3)),
        lambda c: replace(c, execution=replace(c.execution, seed=1337)),
        lambda c: replace(c, engine_source=replace(c.engine_source, digest="a" * 64)),
        lambda c: replace(c, client_source=replace(c.client_source, digest="b" * 64)),
        lambda c: replace(
            c,
            definition_sources={
                **c.definition_sources,
                "model": replace(c.definition_sources["model"], digest="c" * 64),
            },
        ),
    ],
)
def test_saved_bundle_initializer_rebinding_rejects_other_changes(saved_family, change):
    bundle, _ = saved_family
    with pytest.raises(ValueError, match="saved Plans differ"):
        bundle.with_initialization(change(bundle.config))


@pytest.mark.parametrize(
    "change",
    [
        "capability",
        "masked",
        "optimizer",
        "worker",
        "coverage",
        "layout",
        "admission",
        "duplicate",
    ],
)
def test_bundle_rejects_missing_capabilities_and_changed_saved_proofs(
    saved_family, change
):
    bundle, _ = saved_family
    metadata = plain(bundle.metadata)
    plans = {rank: dict(plans) for rank, plans in bundle.plans.items()}
    if change == "capability":
        metadata["candidates"].pop()
    elif change in {"masked", "optimizer"}:
        name = next(
            row["workers"][0][change]
            for row in metadata["candidates"]
            if row["workers"][0][change] is not None
        )
        del plans[0][name]
    elif change == "worker":
        plans[1] = plans.pop(0)
    elif change == "coverage":
        metadata["coverage"][0]["proof"]["max_datums"] += 1
    elif change == "layout":
        metadata["candidates"][0]["layout"]["staging_bytes"] += 1
    elif change == "admission":
        metadata["memory"][0]["execution_bytes"] -= 1
    elif change == "duplicate":
        metadata["candidates"].append(metadata["candidates"][0])
    with pytest.raises(ValueError):
        replace(bundle, plans=plans, metadata=metadata).recheck(installed())


def test_bundle_accepts_missing_optional_coverage_proof(saved_family):
    bundle, _ = saved_family
    metadata = plain(bundle.metadata)
    metadata["coverage"] = []
    restored = replace(bundle, metadata=metadata).recheck(installed())
    assert restored.families and not restored.coverage


def test_bundle_rechecks_source_scenarios_and_rejects_unreferenced_artifacts(
    saved_family,
):
    bundle, _ = saved_family
    key = next(iter(bundle.sources))
    sources = dict(bundle.sources)
    sources[key] = replace(
        sources[key], metadata={**sources[key].metadata, "profiling_scenario": {}}
    )
    with pytest.raises(ValueError, match="policy/scenario"):
        replace(bundle, sources=sources).recheck(installed())
    sources = {**bundle.sources, "a" * 64: bundle.sources[key]}
    with pytest.raises(ValueError, match="referenced source"):
        replace(bundle, sources=sources).recheck(installed())
    plans = {0: {**bundle.plans[0], "b" * 64: next(iter(bundle.plans[0].values()))}}
    with pytest.raises(ValueError, match="referenced worker"):
        replace(bundle, plans=plans).recheck(installed())
    config = replace(
        bundle.config,
        engine_source=replace(bundle.config.engine_source, digest="c" * 64),
    )
    with pytest.raises(ValueError, match="source"):
        replace(bundle, config=config).recheck(installed())


def test_bundle_entry_checksums_duplicates_and_versions_are_strict(saved_family):
    bundle, _ = saved_family
    value = bundle.to_dict()
    value["plans"][0]["value"]["timeline"]["makespan_ns"] += 1
    with pytest.raises(ValueError, match="artifact identity"):
        TrainingBundle.from_dict(value, payloads=bundle._archive_payloads())
    value = bundle.to_dict()
    value["plans"].append(value["plans"][0])
    with pytest.raises(ValueError, match="duplicate bundle Plan"):
        TrainingBundle.from_dict(value, payloads=bundle._archive_payloads())
    value = bundle.to_dict()
    value["sources"].append(value["sources"][0])
    with pytest.raises(ValueError, match="duplicate bundle source"):
        TrainingBundle.from_dict(value, payloads=bundle._archive_payloads())
    for version in (True, 2):
        with pytest.raises(ValueError, match="unsupported training bundle"):
            TrainingBundle.from_dict({**bundle.to_dict(), "version": version})


def test_bundle_raw_load_and_inventory_rejection_need_no_framework(
    saved_family, tmp_path
):
    bundle, _ = saved_family
    # Removing a capability must fail before any native recheck, even in a new
    # process without the test's native fixture substitutions.
    metadata = plain(bundle.metadata)
    metadata["candidates"].pop()
    path = tmp_path / "missing.ssp"
    replace(bundle, metadata=metadata).save(path)
    script = """
import sys
for name in ('torch', 'triton', 'mlops', 'shadowspill_workloads'):
    sys.modules[name] = None
from shadowspill.training.preparation.bundle import TrainingBundle
from engine_tests.training.test_setup_resolution import installed
bundle = TrainingBundle.load(sys.argv[1])
try:
    bundle.recheck(installed())
except ValueError as error:
    assert 'every requested' in str(error), str(error)
else:
    raise AssertionError('incomplete capability family accepted')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
