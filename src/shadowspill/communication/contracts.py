"""Resolve symmetric collective allocation evidence without loading GPU code."""

from shadowspill.artifacts._values import plain
from shadowspill.artifacts.program import Program
from shadowspill.task.allocations import (
    TaskAllocationContract,
    TaskAllocationEvent,
    TaskAllocationOperation,
)
from shadowspill.task.profiles import TaskMeasurement, TaskOutputInputBinding

from .collectives import matching_measurements
from .info import CommunicationInfo
from .memory_records import validate_task_memory
from .task_costs import task_collectives


def collective_measurement(
    program: Program, key: str, info: CommunicationInfo
) -> TaskMeasurement:
    """Every participant must support the same saved physical task layout.

    A workspace total is insufficient: placement and runtime admission need the
    allocating callable's ordered allocation and output-alias records as well.
    """
    records, durations = [], []
    for request in task_collectives(program)[key]:
        rows = matching_measurements(
            info,
            {**request.to_dict(), "setup": program.setup, "topology": info.topology},
        )
        row = next(
            (
                row
                for row in rows
                if row.get("memory", {}).get("memory_kind") == "measured"
            ),
            None,
        )
        if row is None:
            raise ValueError(f"{key}: measured collective allocation evidence missing")
        memory = row["memory"]
        validate_task_memory(memory, request.ranks)
        records.extend(memory["per_rank"])
        durations.append(row["duration_seconds"])
    fields = (
        "workspace_bytes",
        "allocation_trace",
        "allocation_contract",
        "output_input_bindings",
        "off_device_output_leaves",
    )
    baseline = {name: plain(records[0][name]) for name in fields}
    if any({name: plain(row[name]) for name in fields} != baseline for row in records):
        raise ValueError(f"{key}: collective allocation layouts differ between ranks")
    events = tuple(
        TaskAllocationEvent.from_dict(item) for item in baseline["allocation_trace"]
    )
    outputs = {
        event.allocation_ordinal for event in events if event.output_leaf_indices
    }
    # Observation releases its returned tensors. Execution publishes them into
    # the Plan, which owns their subsequent lifetime.
    events = tuple(
        event
        for event in events
        if not (
            event.operation is TaskAllocationOperation.FREE
            and event.allocation_ordinal in outputs
        )
    )
    return TaskMeasurement(
        runtime_ns=round(max(durations) * 1e9),
        workspace_requested_bytes=baseline["workspace_bytes"],
        workspace_charged_bytes=baseline["workspace_bytes"],
        workspace_extent_bytes=(),
        samples_ns=tuple(round(value * 1e9) for value in durations),
        provenance="measured collective callable on every logical participant",
        allocation_trace=events,
        allocation_contract=TaskAllocationContract.from_dict(
            baseline["allocation_contract"]
        ),
        output_input_bindings=tuple(
            TaskOutputInputBinding.from_dict(item)
            for item in baseline["output_input_bindings"]
        ),
        off_device_output_leaves=tuple(baseline["off_device_output_leaves"]),
    )
