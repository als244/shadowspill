"""Communication memory evidence is portable, checked and distinct from timings."""

from copy import deepcopy
from dataclasses import replace

import pytest

from shadowspill import CommunicationInfo, MemoryBudgets, apply_profiles, plan_program
from shadowspill.task.allocations import (
    TaskAllocationContract,
    TaskAllocationEvent,
    TaskAllocationOperation,
)

from ._example import options
from .test_collective_tasks import collective_program, compute_profiles


def measured_info():
    trace = (
        TaskAllocationEvent(0, TaskAllocationOperation.ALLOCATE, 16, 16),
        TaskAllocationEvent(0, TaskAllocationOperation.FREE, 16, 16),
    )
    record = {
        "memory_kind": "measured",
        "workspace_bytes": 16,
        "allocation_trace": [event.to_dict() for event in trace],
        "allocation_contract": TaskAllocationContract.capture(trace).to_dict(),
        "output_input_bindings": [],
        "off_device_output_leaves": [],
    }
    topology = {"node0": (0, 1)}
    setup = {"hardware": "fixture"}
    return CommunicationInfo(
        1e9,
        1e9,
        fixed_device_bytes=96,
        topology=topology,
        metadata={
            "memory_kind": "measured",
            "memory_scope": {
                "groups": [(1, 0)],
                "setup": setup,
                "topology": topology,
                "backend": "nccl",
            },
            "memory_per_rank": [
                {
                    "logical_rank": i,
                    "native_retained_bytes": 32,
                    "external_retained_bytes": 64,
                }
                for i in range(2)
            ],
        },
        measurements=(
            {
                "operation": "reduce_scatter",
                "ranks": (1, 0),
                "tensor_bytes": 32,
                "dtype": "torch.float32",
                "duration_seconds": 1e-6,
                "setup": setup,
                "topology": topology,
                "memory": {
                    "workspace_bytes": 16,
                    "memory_kind": "measured",
                    "per_rank": [{"logical_rank": i, **record} for i in range(2)],
                },
            },
        ),
    )


def test_memory_records_roundtrip(tmp_path):
    info = measured_info()
    info.save(tmp_path / "communication.ssp")
    loaded = CommunicationInfo.load(tmp_path / "communication.ssp")
    assert loaded.to_dict() == info.to_dict()
    with pytest.raises(TypeError):
        loaded.measurements[0]["memory"]["workspace_bytes"] = 0


@pytest.mark.parametrize("damage", ["fixed", "workspace", "trace", "rank", "scope"])
def test_memory_records_reject_understatement_and_mismatch(damage):
    value = deepcopy(measured_info().to_dict())
    memory = value["measurements"][0]["memory"]
    if damage == "fixed":
        value["fixed_device_bytes"] = 95
    elif damage == "workspace":
        memory["workspace_bytes"] = 0
    elif damage == "trace":
        memory["per_rank"][0]["allocation_trace"] = []
    elif damage == "rank":
        memory["per_rank"][0]["logical_rank"] = 1
    else:
        value["metadata"]["memory_scope"]["groups"] = [(0, 2)]
    with pytest.raises(ValueError):
        CommunicationInfo.from_dict(value)


def test_planning_binds_communication_memory_without_compute_remeasurement(tmp_path):
    profiled = apply_profiles(collective_program(), compute_profiles())
    assert profiled.task_profiles["reduce"].workspace_bytes is None
    baseline = measured_info()
    for fixed in (96, 160):
        result = plan_program(
            profiled,
            memory_budgets=MemoryBudgets(512, 1024),
            communication_info=replace(baseline, fixed_device_bytes=fixed),
            search_options=options(),
        )
        assert result.profiled_program.fixed_device_bytes == 0
        assert result.fixed_device_bytes == fixed
        assert result.summary["communication_fixed_bytes"] == fixed
        assert result.summary["execution_peak_bytes"] == fixed + 128
        assert any(
            "external communication scratch" in r
            for r in result.simulation_only_reasons
        )
        result.save(tmp_path / "plan.ssp")
        assert type(result).load(tmp_path / "plan.ssp").summary == result.summary


def test_planning_rejects_missing_or_incompatible_communication_memory():
    profiled = apply_profiles(collective_program(), compute_profiles())
    for info, error in (
        (
            replace(
                measured_info(),
                fixed_device_bytes=96,
                metadata={"memory_kind": "estimated"},
                measurements=(),
                scale_up_bytes_per_second=1e9,
                scale_up_latency_seconds=0,
            ),
            "workspace",
        ),
        (
            replace(measured_info(), metadata={}, fixed_device_bytes=None),
            "fixed_device",
        ),
    ):
        with pytest.raises(ValueError, match=error):
            plan_program(
                profiled,
                memory_budgets=MemoryBudgets(512, 1024),
                communication_info=info,
                search_options=options(),
            )
    value = measured_info().to_dict()
    value["metadata"]["memory_scope"]["setup"] = {"hardware": "different"}
    with pytest.raises(ValueError, match="different groups/setup"):
        plan_program(
            profiled,
            memory_budgets=MemoryBudgets(512, 1024),
            communication_info=CommunicationInfo.from_dict(value),
            search_options=options(),
        )


def test_memory_observation_does_not_override_bandwidth_estimated_timing():
    value = measured_info().to_dict()
    value["measurements"][0]["duration_seconds"] = None
    info = CommunicationInfo.from_dict(value)
    info = replace(info, scale_up_bytes_per_second=1e9, scale_up_latency_seconds=0)
    profiled = apply_profiles(collective_program(), compute_profiles())
    result = profiled._costed_ir(info)
    reduction = next(p for p in result.profiles if p.profile_id == "reduce")
    assert reduction.runtime_ns == 16
    assert reduction.workspace_bytes == 16
