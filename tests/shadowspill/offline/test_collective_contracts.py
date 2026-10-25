"""Shared executable layouts require matching allocation evidence on every rank."""

from dataclasses import replace

import pytest

from shadowspill import CommunicationInfo
from shadowspill.communication.contracts import collective_measurement
from shadowspill.task.allocations import TaskAllocationContract, TaskAllocationEvent

from .test_collective_tasks import collective_program
from .test_communication_memory import measured_info


def test_collective_contract_is_resolved_without_compiled_payload():
    program, info = collective_program(), measured_info()
    observed = collective_measurement(program, "reduce", info)
    assert observed.workspace_charged_bytes == 16
    assert len(observed.allocation_trace) == 2
    assert observed.runtime_ns == 1000


def test_memory_evidence_does_not_require_a_measured_duration():
    value = measured_info().to_dict()
    value["measurements"][0].pop("duration_seconds")
    value.update(scale_up_bytes_per_second=1e9, scale_up_latency_seconds=1e-6)
    observed = collective_measurement(
        collective_program(), "reduce", CommunicationInfo(**value)
    )
    assert observed.runtime_ns == 1016


def test_collective_workspace_total_cannot_stand_in_for_allocation_evidence():
    value = measured_info().to_dict()
    value["measurements"][0]["memory"] = {
        "workspace_bytes": 16,
        "memory_kind": "estimated",
    }
    with pytest.raises(ValueError, match="allocation evidence missing"):
        collective_measurement(
            collective_program(), "reduce", CommunicationInfo(**value)
        )


def test_asymmetric_observations_cannot_share_a_physical_plan():
    value = measured_info().to_dict()
    memory = value["measurements"][0]["memory"]
    row = memory["per_rank"][1]
    trace = tuple(
        replace(TaskAllocationEvent.from_dict(event), charged_bytes=32)
        for event in row["allocation_trace"]
    )
    row.update(
        allocation_trace=[event.to_dict() for event in trace],
        allocation_contract=TaskAllocationContract.capture(trace).to_dict(),
        workspace_bytes=32,
    )
    memory["workspace_bytes"] = 32
    with pytest.raises(ValueError, match="layouts differ between ranks"):
        collective_measurement(
            collective_program(), "reduce", CommunicationInfo(**value)
        )


def test_published_collective_output_remains_live_after_task():
    value = measured_info().to_dict()
    value["measurements"][0]["memory"]["workspace_bytes"] = 0
    for row in value["measurements"][0]["memory"]["per_rank"]:
        first, last = (
            TaskAllocationEvent.from_dict(event) for event in row["allocation_trace"]
        )
        first = replace(first, output_leaf_indices=(0,), output_view_offsets=(0,))
        trace = (first, last)
        row.update(
            workspace_bytes=0,
            allocation_trace=[event.to_dict() for event in trace],
            allocation_contract=TaskAllocationContract.capture(trace).to_dict(),
        )
    observed = collective_measurement(
        collective_program(), "reduce", CommunicationInfo(**value)
    )
    assert len(observed.allocation_trace) == 1
    assert observed.allocation_trace[0].output_leaf_indices == (0,)
