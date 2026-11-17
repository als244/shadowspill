"""Real CPU schedules and admission from synthetic, explicitly static captures."""

import json
import subprocess
import sys
import zipfile
from dataclasses import replace

import pytest

from engine_tests.offline._example import communication
from engine_tests.offline.test_forward_instances import captured_forward
from engine_tests.training.test_allocation_coverage import certify, compiler_fixture
from engine_tests.training.test_measurement_reuse import REUSE
from engine_tests.training.test_preparation_templates import inventory
from engine_tests.training.test_setup_resolution import installed
from shadowspill import Profiles, TaskProfile, apply_profiles
from shadowspill.artifacts._values import plain
from shadowspill.parallelism import InputFieldPlacement, InputPlacement, Parallelism
from shadowspill.task.allocations.contract import TaskAllocationContract
from shadowspill.task.allocations.events import (
    TaskAllocationEvent,
    TaskAllocationOperation,
)
from shadowspill.task.layout import reconcile_compiled_task_layout
from shadowspill.task.profiles import TaskMeasurement
from shadowspill.task.storage import (
    OutputView,
    StorageRoot,
    StorageRootKind,
    TaskStorageContract,
)
from shadowspill.training.definitions import ModelReferenceDefinition
from shadowspill.training.preparation import job
from shadowspill.training.preparation.cost_evidence import (
    communication_edits,
    profile_edits,
)
from shadowspill.training.preparation.layouts import prepare_layouts
from shadowspill.training.preparation.local_planning import PlanningInputs
from shadowspill.training.preparation.local_plans import LocalPlans
from shadowspill.training.preparation.local_reconciliation import reconcile_local_plans
from shadowspill.training.preparation.measurement_export import export_measurements
from shadowspill.training.preparation.setup import resolve_training
from shadowspill.training.preparation.templates import capture_inventory
from shadowspill_client import (
    BatchConfig,
    BuildCandidates,
    Memory,
    PlanningOptions,
    Preparation,
    Weights,
)


@pytest.fixture
def measured(tmp_path, monkeypatch, request):
    ranks = getattr(request, "param", 1)
    ranks, capacity = (ranks, 4 * ranks) if isinstance(ranks, int) else ranks
    fields = inventory().layouts.setup.context.schema.fields
    inv = inventory(
        batch=BatchConfig(limits={"max_examples": capacity}),
        parallelism=Parallelism(
            ranks_per_replica=ranks,
            inputs=InputPlacement.independent(
                ranks, fields=[InputFieldPlacement(tuple(f.path)) for f in fields]
            ),
        ),
        operations={"forward": ["predictions"]},
        prepare=Preparation(
            candidates=BuildCandidates(microbatch_sizes=[4]),
            planning=PlanningOptions(microbatch_counts=[1, 2]),
            external_headroom_bytes=0,  # Static CPU allocation fixture.
        ),
    )
    root = tmp_path / "measured"
    root.mkdir()
    job.PreparationJob((), inv.layouts.setup.config).save(root / job.JOB_NAME)
    monkeypatch.setattr(job, "load_registry", lambda _: installed())
    storage = TaskStorageContract.build(
        (StorageRoot(0, StorageRootKind.FRESH, None, "result", "fixture", 0, 16),),
        tuple(
            OutputView(
                i, 0, 0, size, (size // 4,), (1,), "torch.float32", "torch.strided"
            )
            for i, size in enumerate((16, 8))
        ),
        (),
    )
    trace = (
        TaskAllocationEvent(
            0, TaskAllocationOperation.ALLOCATE, 16, 16, (0, 1), (0, 0)
        ),
    )
    measurement = TaskMeasurement(
        1000, 0, 0, (), (1000,), "static fixture", allocation_trace=trace
    )
    memory = {
        "memory_kind": "static",
        "duration_kind": "measured",
        "allocation_trace": [row.to_dict() for row in trace],
        "allocation_contract": TaskAllocationContract.capture(trace, storage).to_dict(),
        "compiled_layout": reconcile_compiled_task_layout(
            storage, measurement
        ).identity(),
        "output_input_bindings": [],
        "off_device_output_leaves": [],
    }
    for key, template in inv.templates.items():
        original = captured_forward()
        bindings = plain(original.bindings)
        # This CPU fixture uses fully replicated model storage on each worker.
        bindings["weight_shards"] = {}
        bindings.pop("transient_outputs")
        bindings["entrypoints"][0]["storage_handoffs"] = []
        bindings.update(
            {
                "manifests": {
                    "input": {
                        "storage_contract": storage.to_dict(),
                        "root_allocations": [{"root_id": 0, "requested_bytes": 16}],
                    }
                },
                "preparation_template": template.signature(),
                "profiling_scenario": template.signature()["scenario"],
                "compiler_memory": {
                    "input": compiler_fixture(template).bindings["compiler_memory"][
                        "update_impl"
                    ]
                },
            }
        )
        program = certify(
            template,
            replace(
                original,
                tasks=tuple(replace(task, mutations=()) for task in original.tasks),
                bindings=bindings,
                parallelism=inv.layouts.setup.context.parallelism,
            ),
        )
        source = apply_profiles(
            program,
            Profiles(
                tasks={
                    d.profile_key: TaskProfile(
                        1e-6 if d.profile_key == "input" else 0,
                        0,
                        metadata=memory
                        if d.profile_key == "input"
                        else {
                            "memory_kind": "static",
                            "duration_kind": "measured",
                        },
                    )
                    for d in program.task_definitions
                },
                fixed_device_bytes=0,
                metadata={
                    "memory_kind": "static",
                    "preparation_template": key,
                    "profiling_scenario": template.signature()["scenario"],
                    "profiling_options": (
                        inv.layouts.setup.config.preparation.profiling.to_dict()
                    ),
                    "external_headroom_bytes": (
                        inv.layouts.setup.config.preparation.external_headroom_bytes
                    ),
                },
            ),
        )
        program.save(root / "programs" / f"{key}.ssp")
        for stage in ("profiled", "calibrated"):
            source.save(root / stage / f"{key}.ssp")
        communication().save(root / "communication" / f"{key}.ssp")
    path = export_measurements(
        root, tmp_path / "measured.zip", max_disk_bytes=16 << 20, max_files=100
    )
    inputs = PlanningInputs.load(path)
    config = replace(
        inputs.config,
        memory=Memory(3, 1),
        preparation=replace(inputs.config.preparation, reuse=REUSE),
    )
    target = capture_inventory(
        prepare_layouts(resolve_training(installed(), config=config))
    )
    return root, path, inputs, target


def changed_profiles(inputs):
    result = {key: source.profiles for key, source in inputs.sources.items()}
    for profiles in result.values():
        profile = profiles.tasks["input"]
        profiles.tasks["input"] = replace(
            profile,
            duration_seconds=profile.duration_seconds * 3,
            metadata={**profile.metadata, "duration_kind": "estimated"},
        )
    return result


def test_local_planning_can_select_pretrained_weights_without_new_measurements(
    measured,
):
    root, _, inputs, target = measured
    config = target.layouts.setup.config
    source = config.definition_sources["model"]
    registry = installed()
    registry.add_model_reference(
        ModelReferenceDefinition("external/reference@1", lambda _: True, lambda m: m),
        source=source,
    )
    config = replace(
        config,
        model=replace(
            config.model,
            initialize=Weights.pretrained("fixture/model", revision="a" * 40),
        ),
        reference_definition="external/reference@1",
        definition_sources={**config.definition_sources, "reference": source},
    )
    proposal = inputs.plan(config=config)
    inventory = capture_inventory(
        prepare_layouts(resolve_training(registry, config=config))
    )
    bundle, _ = reconcile_local_plans(proposal, root, inventory)
    assert bundle.config.model.initialize == config.model.initialize
    assert bundle.recheck(registry).coverage
    assert all(
        value.program.digest == inputs.sources[key].program.digest
        for key, value in bundle.sources.items()
    )


@pytest.mark.parametrize("measured", [1, 2], indirect=True)
def test_local_cpu_plans_roundtrip_and_reconcile_original_memory_without_binaries(
    measured, tmp_path
):
    root, _, inputs, target = measured
    config = target.layouts.setup.config
    profiles = changed_profiles(inputs)
    comm = {
        key: replace(
            value,
            fetch_bytes_per_second=value.fetch_bytes_per_second / 2,
            metadata={**value.metadata, "timing_kind": "estimated"},
        )
        for key, value in inputs.communication.items()
    }
    proposal = inputs.plan(config=config, profiles=profiles, communication=comm)
    ref = proposal.write(tmp_path / "local.ssp")
    with zipfile.ZipFile(tmp_path / "local.ssp") as archive:
        assert set(archive.namelist()) == {"manifest.json", "artifact.json"}
        assert not json.loads(archive.read("manifest.json"))["blobs"]
    assert ref.media_type == "application/vnd.shadowspill.local-plans+zip"
    restored = LocalPlans.load(tmp_path / "local.ssp")
    bundle, edits = reconcile_local_plans(restored, root, target)
    assert edits and all(
        row["tasks"] and row["communication"] for row in edits.values()
    )
    prepared = bundle.recheck(installed())
    assert prepared.coverage
    for rank, plans in bundle.plans.items():
        assert plans.keys() == bundle.plans[0].keys()
        for candidates in prepared.families.values():
            for row in candidates:
                assert row.workers[rank] == row.workers[0]
    assert not set(proposal.plans) & bundle.plans[0].keys()
    for key, source in bundle.sources.items():
        assert source.program.digest == inputs.sources[key].program.digest
        assert source.fixed_device_bytes == inputs.sources[key].fixed_device_bytes
        assert source.task_profiles["input"].duration_seconds == 3e-6
    assert all(
        source.task_profiles["input"].duration_seconds == 1e-6
        for source in inputs.sources.values()
    )


@pytest.mark.parametrize(
    "damage",
    [
        "workspace",
        "provenance",
        "fixed",
        "kind",
        "task",
        "code",
        "pool",
        "schedule",
        "candidate",
        "missing",
    ],
)
def test_server_rejects_tampered_local_proposals_before_worker_admission(
    measured, damage
):
    root, _, inputs, target = measured
    proposal = inputs.plan(config=target.layouts.setup.config)
    value = proposal.to_dict()
    key = next(iter(value["profiles"]))
    profile = value["profiles"][key]
    plan = next(iter(value["plans"].values()))
    if damage == "workspace":
        profile["tasks"]["input"]["workspace_bytes"] += 1
    elif damage == "provenance":
        profile["tasks"]["input"]["metadata"]["memory_kind"] = "estimated"
    elif damage == "fixed":
        profile["fixed_device_bytes"] += 1
    elif damage == "kind":
        profile["tasks"]["input"]["duration_seconds"] *= 2
    elif damage == "task":
        profile["tasks"]["input"]["compatibility_digest"] = "changed"
    elif damage == "code":
        plan["profiled_program"]["program"]["executables"]["input"] = "a" * 64
    elif damage == "pool":
        plan["profiled_program"]["program"]["alias_groups"][0]["size_bytes"] *= 2
    elif damage == "schedule":
        plan["timeline"]["makespan_ns"] += 1
    elif damage == "candidate":
        value["candidates"] = ["a" * 64]
    elif damage == "missing":
        value["plans"].pop(next(iter(value["plans"])))
    with pytest.raises((ValueError, KeyError)):
        bundle, _ = reconcile_local_plans(LocalPlans.from_dict(value), root, target)
        bundle.recheck(installed())


def test_cpu_planning_needs_no_framework_catalog_or_definition_factory(
    measured, tmp_path
):
    _, path, _inputs, target = measured
    config = tmp_path / "config.json"
    config.write_text(json.dumps(target.layouts.setup.config.to_dict()))
    code = """
import json, sys
for name in ('torch', 'triton', 'mlops', 'shadowspill_workloads'):
    sys.modules[name] = None
from shadowspill.training.preparation.local_planning import PlanningInputs
from shadowspill_client.protocol.training import ResolvedTraining
inputs = PlanningInputs.load(sys.argv[1])
config = ResolvedTraining.from_dict(json.load(open(sys.argv[2])))
proposal = inputs.plan(config=config)
assert proposal.candidates
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(path), str(config)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_local_cost_changes_cannot_relabel_memory_or_collective_geometry(measured):
    _, _, inputs, target = measured
    key = next(iter(inputs.sources))
    source = inputs.sources[key].profiles
    altered = source.merge()
    altered.setup["device"] = "another"
    with pytest.raises(ValueError, match="setup"):
        profile_edits(source, altered)
    original = inputs.communication[key]
    with pytest.raises(ValueError, match="evidence"):
        communication_edits(original, replace(original, topology={"other": (0,)}))
    with pytest.raises(ValueError, match="labeled estimated"):
        communication_edits(original, replace(original, fetch_bytes_per_second=1))
    config = target.layouts.setup.config
    with pytest.raises(ValueError, match="different exported"):
        inputs.plan(
            config=replace(
                config,
                preparation=replace(
                    config.preparation,
                    planning=replace(
                        config.preparation.planning, microbatch_counts=[3]
                    ),
                ),
            )
        )
