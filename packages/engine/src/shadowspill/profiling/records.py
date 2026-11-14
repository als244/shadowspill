"""Validate portable allocation evidence without loading task code or CUDA."""

from collections.abc import Mapping
from typing import Any

from shadowspill.artifacts._values import plain
from shadowspill.artifacts.profiles import TaskProfile
from shadowspill.task.allocations.contract import TaskAllocationContract
from shadowspill.task.allocations.events import (
    TaskAllocationEvent,
    TaskAllocationOperation,
)
from shadowspill.task.layout import reconcile_compiled_task_layout
from shadowspill.task.manifest import ExecutableRootAllocation
from shadowspill.task.profiles import TaskMeasurement, TaskOutputInputBinding
from shadowspill.task.storage import TaskStorageContract


def measured_task(record: TaskProfile, manifest: Mapping[str, Any]) -> TaskMeasurement:
    metadata = record.metadata
    storage = TaskStorageContract.from_dict(plain(manifest["storage_contract"]))
    events = tuple(
        TaskAllocationEvent.from_dict(plain(item))
        for item in metadata["allocation_trace"]
    )
    contract = TaskAllocationContract.from_dict(plain(metadata["allocation_contract"]))
    if TaskAllocationContract.capture(events, storage) != contract:
        raise ValueError("recorded allocation trace differs from its contract")
    outputs = {item.allocation_ordinal for item in events if item.output_leaf_indices}
    temporary = tuple(
        item
        for item in events
        if not (
            item.operation is TaskAllocationOperation.FREE
            and item.allocation_ordinal in outputs
        )
    )
    assert record.duration_seconds is not None and record.workspace_bytes is not None
    result = TaskMeasurement(
        runtime_ns=round(record.duration_seconds * 1e9),
        workspace_requested_bytes=record.workspace_bytes,
        workspace_charged_bytes=record.workspace_bytes,
        workspace_extent_bytes=(),
        samples_ns=(round(record.duration_seconds * 1e9),),
        provenance=str(metadata.get("source", "supplied allocation evidence")),
        allocation_trace=temporary,
        allocation_contract=contract,
        output_input_bindings=tuple(
            TaskOutputInputBinding.from_dict(plain(item))
            for item in metadata["output_input_bindings"]
        ),
        off_device_output_leaves=tuple(metadata["off_device_output_leaves"]),
    )
    layout = reconcile_compiled_task_layout(
        storage,
        result,
        root_allocations=tuple(
            ExecutableRootAllocation(**dict(item))
            for item in manifest["root_allocations"]
        ),
    )
    if layout.identity() != plain(metadata["compiled_layout"]):
        raise ValueError(
            "recorded physical layout differs from its allocation evidence"
        )
    return result
