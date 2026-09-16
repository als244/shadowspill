"""One task invocation recorded by allocation telemetry, and the retention audit."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import torch
from torch.utils._pytree import tree_flatten

from shadowspill.errors import CaptureError
from shadowspill.runtime.telemetry import (
    AllocationTelemetryError,
    TaskWorkspaceProfile,
    read_allocation_telemetry,
    start_allocation_telemetry,
    stop_allocation_telemetry,
    summarize_task_workspace,
)
from shadowspill.task.profiles import TaskOutputInputBinding

from ..executables import ProfileExecutable
from .boundary import AllocatorBoundary


@dataclass(frozen=True, slots=True)
class WorkspaceTimings:
    """Where the wall time of a workspace measurement went."""

    execution_ns: int = 0
    telemetry_copy_decode_ns: int = 0
    replay_ns: int = 0

    def __add__(self, other: WorkspaceTimings) -> WorkspaceTimings:
        return WorkspaceTimings(
            self.execution_ns + other.execution_ns,
            self.telemetry_copy_decode_ns + other.telemetry_copy_decode_ns,
            self.replay_ns + other.replay_ns,
        )

    @property
    def total_ns(self) -> int:
        return self.execution_ns + self.telemetry_copy_decode_ns + self.replay_ns

    def phases(self) -> tuple[tuple[str, int], ...]:
        return (
            ("workspace_execution", self.execution_ns),
            ("telemetry_copy_decode", self.telemetry_copy_decode_ns),
            ("workspace_replay", self.replay_ns),
        )


@dataclass(frozen=True, slots=True)
class WorkspaceObservation:
    """One invocation's workspace profile and what measuring it cost."""

    profile: TaskWorkspaceProfile
    timings: WorkspaceTimings


def measure_workspace(
    boundary: AllocatorBoundary,
    task: Callable[[], object],
    stream: torch.cuda.Stream,
) -> WorkspaceObservation:
    """Measure one task, refusing a measurement the record cannot describe.

    The allocation contract is derived from the recorded events, so a
    truncated record does not describe the task that ran - it describes a
    prefix of it. Re-running the task is not a way out: its scope has
    closed and its outputs have been released. So the only honest
    outcomes are a complete record or an error naming the limit.
    """

    observation = _measure_once(boundary, task, stream)
    if boundary.events_overflowed():
        raise CaptureError(
            "allocation telemetry overflowed at "
            f"{boundary.telemetry_capacity} events, so the recorded "
            "allocations describe only part of this task; raise "
            "telemetry_capacity to profile it"
        )
    return observation


def _measure_once(
    boundary: AllocatorBoundary,
    task: Callable[[], object],
    stream: torch.cuda.Stream,
) -> WorkspaceObservation:
    execution_started = time.perf_counter_ns()
    start_allocation_telemetry(
        boundary.runtime_handle, capacity=boundary.telemetry_capacity
    )
    primary_error: BaseException | None = None
    try:
        with boundary.scope(stream) as task_id:
            output = task()
            output_allocations, output_input_bindings = output_allocation_views(
                boundary,
                output,
                inputs=(
                    task.example_arguments
                    if isinstance(task, ProfileExecutable)
                    else ()
                ),
            )
            # Profiling does not retain task results. Release them while the
            # task range is still active so output-dependent temporary frees
            # remain attributable to this contract. The allocator retires their
            # physical ranges against the active compute stream.
            del output
        boundary.drain(stream, problem="workspace measurement")
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            stop_allocation_telemetry(boundary.runtime_handle)
        except BaseException as cleanup_error:
            if primary_error is None:
                raise
            primary_error.add_note(
                f"allocation telemetry cleanup also failed: {cleanup_error}"
            )
    execution_ns = time.perf_counter_ns() - execution_started
    copy_started = time.perf_counter_ns()
    events = read_allocation_telemetry(boundary.runtime_handle)
    copy_ns = time.perf_counter_ns() - copy_started
    replay_started = time.perf_counter_ns()
    profile = summarize_task_workspace(
        events,
        task_id=task_id,
        output_allocation_views=output_allocations,
        output_input_bindings=output_input_bindings,
    )
    replay_ns = time.perf_counter_ns() - replay_started
    return WorkspaceObservation(
        profile, WorkspaceTimings(execution_ns, copy_ns, replay_ns)
    )


def output_allocation_views(
    boundary: AllocatorBoundary,
    output: object,
    *,
    inputs: Sequence[object] = (),
) -> tuple[
    dict[int, tuple[tuple[int, int], ...]],
    tuple[TaskOutputInputBinding, ...],
]:
    """Map each output leaf to its slab allocation, or to the input it aliases."""

    views_by_allocation: dict[int, list[tuple[int, int]]] = {}
    input_by_allocation: dict[int, int] = {}
    for input_position, value in enumerate(inputs):
        if not isinstance(value, torch.Tensor) or not value.is_cuda:
            continue
        address = value.untyped_storage().data_ptr()
        if address == 0:
            continue
        allocation = boundary.allocation_for_pointer(address)
        input_by_allocation.setdefault(int(allocation.allocation_id), input_position)
    input_bindings: list[TaskOutputInputBinding] = []
    leaves, _ = tree_flatten(output)
    for leaf_index, leaf in enumerate(leaves):
        if not isinstance(leaf, torch.Tensor) or not leaf.is_cuda:
            continue
        address = leaf.untyped_storage().data_ptr()
        if address == 0:
            continue
        allocation = boundary.allocation_for_pointer(address)
        allocation_pointer = int(allocation.pointer or 0)
        offset_bytes = int(leaf.data_ptr()) - allocation_pointer
        if offset_bytes < 0 or offset_bytes > int(allocation.requested_bytes):
            raise CaptureError("compiled output view lies outside its allocator record")
        donated_input_position = input_by_allocation.get(int(allocation.allocation_id))
        if donated_input_position is not None:
            input_bindings.append(
                TaskOutputInputBinding(leaf_index, donated_input_position, offset_bytes)
            )
        else:
            views_by_allocation.setdefault(allocation.allocation_id, []).append(
                (leaf_index, offset_bytes)
            )
    return (
        {
            allocation_id: tuple(views)
            for allocation_id, views in views_by_allocation.items()
        },
        tuple(input_bindings),
    )


def audit_workspace_retention(
    observe: Callable[[], WorkspaceObservation],
    requested_allocated_bytes: Callable[[], int],
    *,
    maximum_iterations: int = 16,
) -> WorkspaceObservation:
    """Distinguish bounded provider caches from unbounded task leakage.

    A task whose trace shows still-live allocations is measured again until
    the process's live bytes stop moving between invocations: a provider
    cache fills once and stays, a leak keeps growing. The returned
    observation carries the timings of every measurement it took.
    """

    previous = requested_allocated_bytes()
    timings = WorkspaceTimings()
    for _ in range(maximum_iterations):
        observation = observe()
        timings += observation.timings
        current = requested_allocated_bytes()
        if not observation.profile.persistent_extent_bytes or current == previous:
            return replace(observation, timings=timings)
        previous = current
    raise AllocationTelemetryError(
        "task retains anonymous allocations without reaching a bounded "
        f"live-byte baseline after {maximum_iterations} invocations; "
        f"latest={observation.profile.persistent_extent_bytes}"
    )
