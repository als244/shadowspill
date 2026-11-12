"""Task reuse preserves implementation, physical evidence and rank agreement."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from shadowspill import Profiles, TaskProfile, TaskProfileCache
from shadowspill.artifacts import TaskDefinition
from shadowspill.artifacts._values import digest
from shadowspill.profiling.cache_identity import task_identities
from shadowspill.profiling.session import MeasurementSession
from shadowspill.task.allocations.contract import TaskAllocationContract
from shadowspill.task.layout import reconcile_compiled_task_layout
from shadowspill.task.profiles import TaskMeasurement
from shadowspill.task.profiling import ProfilingOptions
from shadowspill.task.storage import TaskStorageContract


@pytest.fixture
def evidence():
    storage = TaskStorageContract.build((), (), ())
    contract = TaskAllocationContract.capture((), storage)
    measurement = TaskMeasurement(
        runtime_ns=1000,
        workspace_requested_bytes=0,
        workspace_charged_bytes=0,
        workspace_extent_bytes=(),
        samples_ns=(1000,),
        provenance="fixture",
        allocation_trace=(),
        allocation_contract=contract,
    )
    manifest = {"storage_contract": storage.to_dict(), "root_allocations": []}
    profile = TaskProfile(
        1e-6,
        0,
        "implementation-v1",
        metadata={
            "duration_kind": "measured",
            "memory_kind": "measured",
            "allocation_trace": [],
            "allocation_contract": contract.to_dict(),
            "output_input_bindings": [],
            "off_device_output_leaves": [],
            "compiled_layout": reconcile_compiled_task_layout(
                storage, measurement
            ).identity(),
            "samples_seconds": [1e-6] * 5,
            "warmup_persistent_bytes": [],
        },
    )
    return manifest, profile


def test_persistent_reuse_validates_evidence_and_sampling(tmp_path, evidence):
    manifest, profile = evidence
    identity = {"implementation": profile.compatibility_digest}
    cache = TaskProfileCache(tmp_path)
    assert cache.get(identity, manifest, minimum_samples=3) is None
    assert cache.put(identity, manifest, profile, 128)
    cold = TaskProfileCache(tmp_path)
    hit = cold.get(identity, manifest, minimum_samples=3)
    assert hit.profile == profile and hit.external_device_bytes == 128
    assert cold.get(identity, manifest, minimum_samples=6) is None
    extended = replace(
        profile, metadata={**profile.metadata, "samples_seconds": [1e-6] * 6}
    )
    cold.put(identity, manifest, extended, 128)
    assert (
        TaskProfileCache(tmp_path).get(identity, manifest, minimum_samples=6)
        is not None
    )
    assert (
        cold.get({"implementation": "different kernel"}, manifest, minimum_samples=3)
        is None
    )
    path = next(cold.store.task_profiles.rglob("measurement.json"))
    data = json.loads(path.read_text())
    data["measurement"]["profile"]["workspace_bytes"] = 10
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="corrupt"):
        TaskProfileCache(tmp_path).get(identity, manifest, minimum_samples=3)


def test_refresh_reuses_new_sweep_measurements_and_read_only_never_writes(
    tmp_path, evidence
):
    manifest, profile = evidence
    identity = {"implementation": profile.compatibility_digest}
    old = TaskProfileCache(tmp_path)
    old.put(identity, manifest, profile, 100)
    refresh = TaskProfileCache(tmp_path, mode="refresh")
    assert refresh.get(identity, manifest, minimum_samples=3) is None
    refresh.put(identity, manifest, replace(profile, duration_seconds=2e-6), 200)
    assert (
        refresh.get(identity, manifest, minimum_samples=3).external_device_bytes == 200
    )
    assert (
        TaskProfileCache(tmp_path)
        .get(identity, manifest, minimum_samples=3)
        .profile.duration_seconds
        == 2e-6
    )
    read_only = TaskProfileCache(tmp_path / "read-only", mode="reuse")
    read_only.put(identity, manifest, profile, 100)
    assert read_only.get(identity, manifest, minimum_samples=3) is not None
    assert not read_only.store.task_profiles.exists()


def test_lazy_allocations_unknown_fixed_memory_and_invalid_layout_are_not_cached(
    tmp_path, evidence
):
    manifest, profile = evidence
    identity = {"implementation": profile.compatibility_digest}
    cache = TaskProfileCache(tmp_path)
    retained = replace(
        profile, metadata={**profile.metadata, "warmup_persistent_bytes": [256]}
    )
    assert not cache.put(identity, manifest, retained, 100)
    assert not cache.put(identity, manifest, profile, None)
    invalid = replace(profile, metadata={**profile.metadata, "compiled_layout": {}})
    with pytest.raises(ValueError, match="physical layout differs"):
        cache.put(identity, manifest, invalid, 100)
    assert not cache.store.task_profiles.exists()


def fixture_program(manifest):
    component = {"profile_key": "task", "inputs": [], "outputs": []}
    program = SimpleNamespace(
        task_definitions=(TaskDefinition("task", "implementation-v1"),),
        bindings={
            "training_templates": [[]],
            "optimizer_templates": [component],
            "task_inputs": {"task": {"count": 0}},
            "manifests": {"task": manifest},
        },
        setup={"gpu": "fixture"},
        resources={},
    )
    return program, [{"optimizer": component}]


def test_identity_uses_finalized_code_inputs_and_policy_but_not_capture_timer(evidence):
    manifest, _ = evidence
    program, jobs = fixture_program(manifest)
    options = ProfilingOptions()

    def key():
        return digest(task_identities(program, jobs, options=options, environment={}))

    baseline = key()
    program.bindings["manifests"]["task"]["contract_capture_ns"] = 100
    assert key() == baseline
    options = replace(options, minimum_samples=5)
    assert key() == baseline
    program.task_definitions = (TaskDefinition("task", "different-selected-kernel"),)
    assert key() != baseline
    program.task_definitions = (TaskDefinition("task", "implementation-v1"),)
    program.bindings["task_inputs"]["task"]["static_values"] = [[0, 1]]
    assert key() != baseline


@pytest.mark.parametrize("peer_has_hit", [False, True])
def test_collective_reuse_intersects_hits_before_restoring_tasks(
    tmp_path, evidence, peer_has_hit
):
    manifest, profile = evidence
    program, jobs = fixture_program(manifest)
    cache = TaskProfileCache(tmp_path)
    restored = []
    groups = SimpleNamespace(
        logical_rank=0,
        topology={"node0": [0, 1]},
        assignment=SimpleNamespace(logical_by_worker=(0, 1)),
        check=lambda error: error is None or pytest.fail(error),
        gather=lambda hits: [
            hits,
            hits if isinstance(hits, bool) or peer_has_hit else [],
        ],
    )
    # Exercise the actual selection logic without a GPU/context constructor.
    session = object.__new__(MeasurementSession)
    session.program, session.cache, session.groups = program, cache, groups
    session.cache_allowed = True
    session.options = ProfilingOptions()
    session.streams = SimpleNamespace(description=[])
    session.profiles, session.external = Profiles(), 10
    session.cache_hits, session.cache_misses = [], []
    session.function = restored.append
    session.reuse(jobs)
    cache.put(session.cache_identities["task"], manifest, profile, 1234)
    session.reuse(jobs)
    assert restored == (["task"] if peer_has_hit else [])
    assert list(session.profiles.tasks) == restored
    assert session.external == (1234 if peer_has_hit else 10)
    assert session.cache_misses == ([] if peer_has_hit else ["task"])


def test_peer_input_override_disables_reuse_on_every_rank(tmp_path):
    session = object.__new__(MeasurementSession)
    session.cache, session.cache_allowed = TaskProfileCache(tmp_path), True
    session.groups = SimpleNamespace(
        gather=lambda enabled: [enabled, False],
        check=lambda error: error is None or pytest.fail(error),
    )
    session.reuse([])
    assert session.cache is None
