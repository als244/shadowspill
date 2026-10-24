"""Warm, time and validate one finalized task using the ordinary allocator."""

from collections.abc import Callable, Mapping, Sequence
from statistics import median
from typing import Any

import torch

from shadowspill.artifacts._values import plain
from shadowspill.artifacts.profiles import TaskProfile
from shadowspill.errors import CaptureError
from shadowspill.task.allocations import TaskAllocationContract, TaskAllocationOperation
from shadowspill.task.layout import reconcile_compiled_task_layout
from shadowspill.task.manifest import ExecutableRootAllocation
from shadowspill.task.profiles import TaskMeasurement
from shadowspill.task.profiling import ProfilingOptions
from shadowspill.task.storage import TaskStorageContract

from .event_timing import EventTimer
from .observation.observe import observe_task
from .timing import collect_timing_samples, condition_task


def measure_task(
    function: Callable[..., Any],
    arguments: Sequence[Any],
    manifest: Mapping[str, Any],
    *,
    device: torch.device,
    warmup_seconds: float,
    measure_seconds: float,
) -> TaskProfile:
    cold = observe_task(function, arguments, device=device)
    timer = EventTimer(lambda: function(*arguments))
    options = ProfilingOptions(
        conditioning_seconds=warmup_seconds, measurement_seconds=measure_seconds
    )
    condition_task(timer.sample, options=options)
    observations = [observe_task(function, arguments, device=device) for _ in range(3)]
    storage = TaskStorageContract.from_dict(plain(manifest["storage_contract"]))
    contracts = [
        TaskAllocationContract.capture(item.trace, storage) for item in observations
    ]
    if len({item.compatibility_digest for item in contracts}) != 1:
        raise CaptureError(
            "allocation requests change across repeated warmed executions"
        )
    if any(item.persistent_bytes for item in observations):
        raise CaptureError(
            "warmed task retains anonymous allocations; resource setup is unresolved"
        )
    baseline = observations[0]
    if any(item.input_bindings != baseline.input_bindings for item in observations[1:]):
        raise CaptureError(
            "output/input aliases change across repeated warmed executions"
        )
    timing = collect_timing_samples(timer.sample, options=options)
    samples = tuple(sample / 1e9 for sample in timing.samples)
    duration = median(samples)
    output_ordinals = {
        event.allocation_ordinal
        for event in baseline.trace
        if event.output_leaf_indices
    }
    temporary_trace = tuple(
        event
        for event in baseline.trace
        if not (
            event.operation is TaskAllocationOperation.FREE
            and event.allocation_ordinal in output_ordinals
        )
    )
    measurement = TaskMeasurement(
        runtime_ns=round(duration * 1e9),
        workspace_requested_bytes=baseline.workspace_requested_bytes,
        workspace_charged_bytes=baseline.workspace_bytes,
        workspace_extent_bytes=(),
        samples_ns=tuple(round(item * 1e9) for item in samples),
        provenance="native-allocator-observer+cuda-events",
        allocation_trace=temporary_trace,
        output_input_bindings=baseline.input_bindings,
        allocation_contract=contracts[0],
        off_device_output_leaves=baseline.off_device_outputs,
    )
    layout = reconcile_compiled_task_layout(
        storage,
        measurement,
        root_allocations=tuple(
            ExecutableRootAllocation(**dict(value))
            for value in manifest["root_allocations"]
        ),
    )
    return TaskProfile(
        duration_seconds=duration,
        workspace_bytes=baseline.workspace_bytes,
        metadata={
            "duration_kind": "measured",
            "memory_kind": "measured",
            "source": "native-allocator-observer+cuda-events",
            "allocation_trace": [item.to_dict() for item in baseline.trace],
            "allocation_contract": contracts[0].to_dict(),
            "output_input_bindings": [
                item.to_dict() for item in baseline.input_bindings
            ],
            "off_device_output_leaves": baseline.off_device_outputs,
            "compiled_layout": layout.identity(),
            "samples_seconds": samples,
            "timing_relative_mad": timing.relative_mad,
            "timing_half_drift": timing.half_drift,
            "timing_unstable": timing.unstable(options),
            "sampling": timing.window.to_dict(),
            "warmup_persistent_bytes": cold.persistent_bytes,
            "coverage": "observed paths; physical execution admission is separate",
        },
    )
