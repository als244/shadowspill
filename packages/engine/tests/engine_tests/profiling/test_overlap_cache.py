"""Paired reuse must never skip unmatched collectives or supply memory evidence."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from shadowspill import OverlapProfileCache, Profiles, TaskProfile
from shadowspill.artifacts import TaskDefinition
from shadowspill.profiling.overlap import _OverlapSession


def observation():
    return {
        "timing_class": "collective_test",
        "bytes": 1024,
        "status": "measured",
        "per_rank": [
            {
                "logical_rank": rank,
                "rates": [0.8, 0.5],
                "requested_trials": 1,
                "trials": {mode: [{}] for mode in ("task", "transfer", "both")},
            }
            for rank in range(2)
        ],
    }


def identity():
    return {
        "task": {
            "digest": "finalized-task-and-inputs",
            "environment": {
                "rank_assignment": [0, 1],
            },
        },
        "route": {"timing_class": "collective_test", "bytes": 1024},
        "trials": None,
    }


def test_paired_cache_integrity_route_identity_and_no_negative_entries(tmp_path):
    cache, request, row = OverlapProfileCache(tmp_path), identity(), observation()
    assert cache.get(request) is None
    assert cache.put(request, row)
    assert OverlapProfileCache(tmp_path).get(request) == row
    changed = deepcopy(request)
    changed["route"]["bytes"] *= 2
    assert cache.get(changed) is None
    assert not cache.put(request, {**row, "status": "insufficient_device_memory"})
    with pytest.raises(ValueError, match="compatible supported"):
        cache.put(changed, row)
    damaged = deepcopy(row)
    damaged["per_rank"] = damaged["per_rank"][:1]
    with pytest.raises(ValueError, match="compatible supported"):
        cache.put(request, damaged)
    path = next(cache.store.overlap_profiles.rglob("measurement.json"))
    data = json.loads(path.read_text())
    data["observation"]["bytes"] *= 2
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="corrupt"):
        OverlapProfileCache(tmp_path).get(request)


def test_refresh_reuses_this_sweep_and_read_only_never_persists(tmp_path):
    request, row = identity(), observation()
    OverlapProfileCache(tmp_path).put(request, row)
    cache = OverlapProfileCache(tmp_path, mode="refresh")
    assert cache.get(request) is None
    row["per_rank"][0]["rates"][0] = 0.7
    cache.put(request, row)
    assert cache.get(request) == row
    assert OverlapProfileCache(tmp_path).get(request) == row
    read_only = OverlapProfileCache(tmp_path / "reader", mode="reuse")
    read_only.put(request, row)
    assert read_only.get(request) == row
    assert not read_only.store.overlap_profiles.exists()
    # Returned evidence cannot mutate a subsequent cache hit.
    read_only.get(request)["per_rank"].clear()
    assert read_only.get(request) == row


def session(tmp_path, *, peer="matching", mode="contribute"):
    component = {"profile_key": "task", "inputs": [], "outputs": []}
    program = SimpleNamespace(
        task_definitions=(TaskDefinition("task", "implementation-v1"),),
        bindings={
            "training_templates": [[]],
            "optimizer_templates": [component],
            "task_inputs": {"task": {"count": 0}},
            "manifests": {"task": {}},
        },
        setup={"gpu": "fixture"},
        resources={},
    )

    def gather(value):
        if type(value) is bool:
            return [value, False if peer == "explicit-inputs" else value]
        if isinstance(value, list) and value:
            if peer == "missing":
                return [value, []]
            if peer == "different-evidence":
                return [value, [(item, "another-digest") for item, _ in value]]
        return [value, value]

    def check(error):
        if error is not None:
            raise ValueError(error)

    group = SimpleNamespace(
        logical_rank=0,
        topology={"node0": [0, 1]},
        assignment=SimpleNamespace(logical_by_worker=(0, 1)),
        gather=gather,
        check=check,
    )
    result = object.__new__(_OverlapSession)
    result.program, result.groups, result.inputs = program, group, {}
    result.overlap_cache = OverlapProfileCache(tmp_path, mode=mode)
    result.streams = SimpleNamespace(description=[])
    result.trials = None
    result.routes = (
        SimpleNamespace(timing_class="collective_test", bytes=1024, requests=()),
    )
    result.baseline = {"task": TaskProfile(compatibility_digest="implementation-v1")}
    result.route_identities, result.cached_routes = {}, {}
    result.cache_hits, result.cache_misses, result.profiles = [], [], Profiles()
    return result, [{"optimizer": component}]


@pytest.mark.parametrize("peer", ["matching", "missing", "different-evidence"])
def test_ranks_agree_on_identical_evidence_before_skipping_entire_tasks(tmp_path, peer):
    value, jobs = session(tmp_path, peer=peer)
    value.reuse_overlaps(jobs, {"backend": "fixture"})
    request = next(iter(value.route_identities.values()))
    value.overlap_cache.put(request, observation())
    value.cache_hits, value.cache_misses = [], []
    value.reuse_overlaps(jobs, {"backend": "fixture"})
    if peer == "matching":
        profile = value.profiles.tasks["task"]
        assert len(value.cache_hits) == 1 and not value.cache_misses
        assert profile.duration_seconds is None and profile.workspace_bytes is None
        assert profile.metadata["overlap_calibration"]["status"] == "cached"
        assert len(profile.overlap_costs) == 1
    else:
        assert not value.profiles.tasks and not value.cache_hits
        assert len(value.cache_misses) == 1


@pytest.mark.parametrize(
    "change", ["kernel", "inputs", "backend", "trials", "rank-map"]
)
def test_reuse_identity_tracks_execution_and_measurement_contract(tmp_path, change):
    value, jobs = session(tmp_path)
    value.reuse_overlaps(jobs, {})
    request = next(iter(value.route_identities.values()))
    value.overlap_cache.put(request, observation())
    if change == "kernel":
        value.program.task_definitions = (TaskDefinition("task", "other-kernel"),)
    elif change == "inputs":
        value.program.bindings["task_inputs"]["task"]["static_values"] = [1]
    elif change == "trials":
        value.trials = 5
    elif change == "rank-map":
        value.groups.assignment.logical_by_worker = (1, 0)
    value.cache_hits, value.cache_misses = [], []
    value.reuse_overlaps(jobs, {"backend": "other"} if change == "backend" else {})
    assert not value.cache_hits and len(value.cache_misses) == 1


def test_explicit_inputs_disable_all_rank_reuse_and_require_refuses_misses(tmp_path):
    value, jobs = session(tmp_path, peer="explicit-inputs")
    value.reuse_overlaps(jobs, {})
    assert value.overlap_cache is None
    value, jobs = session(tmp_path, mode="require")
    with pytest.raises(ValueError, match="refuses to produce"):
        value.reuse_overlaps(jobs, {})
