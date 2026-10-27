"""Small saved compute/memory evidence for CPU corpus and planner tests."""

from dataclasses import replace

from shadowspill import apply_profiles
from shadowspill.task.allocations.contract import TaskAllocationContract
from shadowspill.task.allocations.events import (
    TaskAllocationEvent,
    TaskAllocationOperation,
)
from shadowspill.task.layout import reconcile_compiled_task_layout
from shadowspill.task.profiles import TaskMeasurement
from shadowspill.task.storage import TaskStorageContract
from tests.shadowspill.offline._example import profiled


def measured(original=None):
    original = profiled() if original is None else original
    profiles = original.profiles
    storage = TaskStorageContract.build((), (), ())
    for key, record in profiles.tasks.items():
        size = record.workspace_bytes
        trace = tuple(
            TaskAllocationEvent(0, operation, size, size)
            for operation in (
                TaskAllocationOperation.ALLOCATE,
                TaskAllocationOperation.FREE,
            )
            if size
        )
        measurement = TaskMeasurement(
            1000, size, size, (), (1000,), "fixture", allocation_trace=trace
        )
        profiles.tasks[key] = replace(
            record,
            metadata={
                "duration_kind": "static",
                "memory_kind": "static",
                "allocation_trace": [event.to_dict() for event in trace],
                "allocation_contract": TaskAllocationContract.capture(
                    trace, storage
                ).to_dict(),
                "compiled_layout": reconcile_compiled_task_layout(
                    storage, measurement
                ).identity(),
                "output_input_bindings": [],
                "off_device_output_leaves": [],
            },
        )
    profiles.metadata["memory_kind"] = "static"
    program = replace(
        original.program,
        executables={key: b"not executed" for key in profiles.tasks},
        bindings={
            "manifests": {
                key: {"storage_contract": storage.to_dict(), "root_allocations": []}
                for key in profiles.tasks
            },
            "entrypoints": [
                {"task_id": task.task_id, "input_slots": [], "output_slots": []}
                for task in original.program.tasks
            ],
        },
    )
    return apply_profiles(program, profiles)
