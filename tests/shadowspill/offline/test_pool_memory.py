"""Resource evidence changes physical allowance, not the declared budget."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest

from shadowspill import (
    MemoryBudgets,
    Plan,
    PoolMemoryProfile,
    plan_program,
    sweep_programs,
)
from shadowspill.errors import PlanInfeasibleError
from shadowspill.planning.worker_memory import (
    pool_process_reservation,
    shared_fixed_overhead,
    shared_pool_memory,
)

from ._example import communication, options, profiled


def evidence(*, total=1000, global_bytes=16, rank=0, uuid="GPU-a", capacity=64):
    def sample(process, outside):
        return {
            "physical": {
                "process_bytes": process,
                "device_used_bytes": process + outside,
                "device_total_bytes": total,
            },
            "other_processes": [],
            "own_process_present": True,
            "own_process_memory_available": True,
        }

    cycle = {
        "before": [sample(10, 20)] * 2,
        "live": [sample(16, 20 + global_bytes)] * 2,
        "warmed": [sample(16, 20 + global_bytes)] * 2,
        "released": [sample(12, 20)] * 2,
    }
    return PoolMemoryProfile(
        {
            "version": 1,
            "kind": 1,
            "kind_name": "pinned_host",
            "capacity": capacity,
            "configuration": {},
            "numa_binding": True,
        },
        (
            {
                "logical_rank": rank,
                "identity": {
                    "gpu_uuid": uuid,
                    "gpu_name": "test GPU",
                    "driver": "d",
                    "host": "h",
                    "libraries": {"backend": "abc"},
                },
                "execution_control_bytes": 4,
                "method": "full_capacity_pool_and_routes_v1",
                "startup_peak_bytes": None,
                "cycles": [deepcopy(cycle), deepcopy(cycle)],
            },
        ),
    )


def test_profile_roundtrip_keeps_raw_unknowns_and_separate_scopes(tmp_path):
    profile = evidence()
    profile.save(tmp_path / "pool.ssp")
    loaded = PoolMemoryProfile.load(tmp_path / "pool.ssp")
    assert loaded == profile
    assert loaded.physical_limit_bytes == 964
    assert loaded.process_reservation_bytes == 2
    assert loaded.global_reservation_bytes(0) == 16
    assert loaded.ranks[0]["startup_peak_bytes"] is None
    assert loaded.resource_identity == profile.resource_identity


@pytest.mark.parametrize(
    "updates", [{"gpu_uuid": "GPU-b", "host": "h2"}, {"driver": "new-driver"}]
)
def test_worker_reuses_calibration_when_only_provenance_changes(updates):
    profile = evidence()
    original = profile.to_dict()
    identity = {**original["ranks"][0]["identity"], **updates}
    profile.validate_worker(0, identity)
    assert profile.to_dict() == original
    assert profile.physical_limit_bytes == 964
    assert profile.global_reservation_bytes(0) == 16


@pytest.mark.parametrize("field", ["gpu_name", "libraries"])
def test_worker_still_requires_compatible_pool_hardware_and_software(field):
    profile = evidence()
    identity = profile.to_dict()["ranks"][0]["identity"]
    identity[field] = {"backend": "changed"} if field == "libraries" else "changed"
    with pytest.raises(ValueError, match=field):
        profile.validate_worker(0, identity)


def test_process_budget_stays_independent_and_global_is_not_double_charged(tmp_path):
    profile = evidence(total=200, global_bytes=16)  # Physical allowance=164.
    plan = plan_program(
        profiled(),
        memory_budgets=MemoryBudgets(200, 64),
        communication_info=communication(),
        search_options=options(),
        pool_memory=profile,
    )
    assert plan.memory_budgets.execution_bytes == 200
    assert plan.effective_execution_bytes == 164
    assert plan.fixed_device_bytes == 22
    plan.save(tmp_path / "plan.ssp")
    loaded = Plan.load(tmp_path / "plan.ssp")
    assert loaded.to_dict() == plan.to_dict()
    low = plan_program(
        profiled(),
        memory_budgets=MemoryBudgets(120, 64),
        communication_info=communication(),
        search_options=options(),
        pool_memory=profile,
    )
    assert low.effective_execution_bytes == 120
    assert low.fixed_device_bytes == 22


def test_calibrated_capacity_rejects_a_plan_that_process_only_budget_admits():
    with pytest.raises(PlanInfeasibleError):
        plan_program(
            profiled(),
            memory_budgets=MemoryBudgets(200, 64),
            communication_info=communication(),
            search_options=options(),
            pool_memory=evidence(total=136),
        )


def test_candidate_sweep_reuses_one_profile_and_preserves_refusals():
    result = sweep_programs(
        {"one": profiled(), "two": profiled()},
        memory_budgets=[MemoryBudgets(100, 64), MemoryBudgets(120, 64)],
        communication_info=communication(),
        search_options=options(),
        pool_memory=evidence(),
    )
    assert len(result.cases) == 4
    assert all(c.failure is not None for c in result.cases[:2])
    assert all(c.plan.pool_memory == evidence() for c in result.cases[2:])


def test_shared_worker_reserves_pool_once_and_refuses_different_evidence():
    plan = plan_program(
        profiled(),
        memory_budgets=MemoryBudgets(200, 64),
        communication_info=communication(),
        search_options=options(),
        pool_memory=evidence(),
    )
    assert pool_process_reservation([plan] * 3) == 2
    assert shared_fixed_overhead([plan] * 3) == 22
    with pytest.raises(ValueError, match="identical pool memory"):
        shared_pool_memory([plan, replace(plan, pool_memory=None)])
    with pytest.raises(ValueError, match="identical pool memory"):
        shared_pool_memory([plan, replace(plan, pool_memory=evidence(total=999))])


def test_dp_requires_complete_distinct_device_evidence():
    first = evidence()
    second = evidence(total=800, global_bytes=32, rank=1, uuid="GPU-b")
    combined = PoolMemoryProfile.combine(first, second)
    combined.validate_program(64, 2)
    assert combined.physical_limit_bytes == 748
    assert combined.global_reservation_bytes(1) == 32
    with pytest.raises(ValueError, match="distinct ranks and physical GPUs"):
        PoolMemoryProfile.combine(first, evidence(rank=1))
    with pytest.raises(ValueError, match="logical ranks"):
        first.validate_program(64, 2)
    with pytest.raises(ValueError, match="spill budget"):
        first.validate_program(65, 1)
    with pytest.raises(ValueError, match="Program target"):
        first.validate_program(64, 1, gpu="another GPU")
    with pytest.raises(ValueError, match="communication timing"):
        first.validate_program(64, 1, spill_kind="ssd")


@pytest.mark.parametrize(
    "mutation",
    [
        "other_process",
        "baseline",
        "total",
        "peak",
        "missing_samples",
        "missing_counter",
    ],
)
def test_unresolved_measurements_are_never_converted_to_zero(mutation):
    data = evidence().to_dict()
    rank = data["ranks"][0]
    if mutation == "other_process":
        rank["cycles"][0]["live"][0]["other_processes"] = [123]
    elif mutation == "baseline":
        rank["cycles"][0]["released"][0]["physical"]["device_used_bytes"] += 1
    elif mutation == "total":
        rank["cycles"][0]["live"][0]["physical"]["device_total_bytes"] = 0
    elif mutation == "missing_counter":
        rank["cycles"][0]["live"][0]["own_process_memory_available"] = False
    elif mutation == "peak":
        rank["startup_peak_bytes"] = 0
    else:
        rank["cycles"][0]["live"] = []
    with pytest.raises(ValueError):
        PoolMemoryProfile.from_dict(data)


def test_old_plan_archive_does_not_invent_pool_measurement():
    plan = plan_program(
        profiled(),
        memory_budgets=MemoryBudgets(200, 64),
        communication_info=communication(),
        search_options=options(),
    )
    data = plan.to_dict()
    del data["pool_memory"]
    loaded = Plan.from_dict(json.loads(json.dumps(data)))
    assert loaded.pool_memory is None
    assert loaded.effective_execution_bytes == 200
