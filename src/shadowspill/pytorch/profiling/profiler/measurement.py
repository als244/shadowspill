"""One task measured end to end: warmed, probed, timed, traced, assembled."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from shadowspill.profiling.timing import (
    STABLE_VARIABILITY,
    TimingObservation,
    collect_timing_samples,
)
from shadowspill.runtime.telemetry import (
    AllocationTelemetryError,
    TaskWorkspaceProfile,
)
from shadowspill.task.allocations import (
    TaskAllocationContract,
    TaskAllocationPathObservation,
)
from shadowspill.task.profiles import TaskMeasurement

from ..executables import ProfileExecutable
from .contract import ProbedPath, probe_allocation_paths, validate_allocation_contract
from .workspace import (
    WorkspaceObservation,
    audit_workspace_retention,
    measure_workspace,
)

if TYPE_CHECKING:
    from . import TaskProfiler

#: Invocations allowed for a provider's allocations to settle after warmup.
STABILIZATION_BUDGET = 16


class MeasuredTask:
    """The callable under measurement, opened once per invocation.

    This form is the callable itself, which every invocation reuses and the
    path probes may rebind.  A task whose state exists only once overrides
    ``open`` to build a fresh one each time.
    """

    def __init__(self, task: Callable[[], object]) -> None:
        self.task = task

    @contextmanager
    def open(self) -> Iterator[Callable[[], object]]:
        yield self.task


def measure_task(
    profiler: TaskProfiler,
    source: MeasuredTask,
    *,
    execution_provider: str,
) -> TaskMeasurement:
    """Warm, time and trace one task through the allocator boundary.

    ``source`` opens the callable for every invocation, which is what lets a
    first optimizer step -- whose state exists only once -- be measured the
    same way as a compiled graph that can simply be called again.
    """

    boundary = profiler.boundary
    stream = boundary.stream()
    phases: list[tuple[str, int]] = []

    def invoke() -> None:
        with source.open() as task:
            boundary.invoke(task, stream)

    def sample() -> int:
        with source.open() as task:
            return boundary.time_once(task, stream)

    def observe(task: Callable[[], object]) -> WorkspaceObservation:
        return measure_workspace(boundary, task, stream)

    def observe_current() -> WorkspaceObservation:
        with source.open() as task:
            return observe(task)

    try:
        if not boundary.conditioned:
            with timed(phases, "device_conditioning"):
                boundary.condition_device(stream)
        # Provider compilation, autotuning, and shape-keyed initialization are
        # planning-time setup, not alternative task allocation paths. Warm them
        # before varying representative input identities.
        with timed(phases, "provider_warmup"):
            warm_provider(invoke, boundary.requested_allocated_bytes, profiler.warmups)
        contract = None
        path_probes: tuple[ProbedPath, ...] = ()
        if isinstance(source.task, ProfileExecutable):
            contract = source.task.artifact.storage_contract
            with timed(phases, "allocation_path_probes"):
                source.task, path_probes = probe_allocation_paths(
                    boundary,
                    stream,
                    profiler.executables,
                    source.task,
                    observe,
                    seeds=profiler.probe_seeds,
                    repetitions=profiler.probe_repetitions,
                )
            with timed(phases, "post_probe_stabilization"):
                warm_provider(invoke, boundary.requested_allocated_bytes, 1)
        with timed(phases, "timing_samples"):
            timing = collect_timing_samples(sample, minimum=profiler.samples)
        boundary.require_idle(problem="timing measurement")
        audited = time.perf_counter_ns()
        observation = audit_workspace_retention(
            observe_current, boundary.requested_allocated_bytes
        )
        phases.extend(observation.timings.phases())
        elapsed = time.perf_counter_ns() - audited - observation.timings.total_ns
        phases.append(("retention_audit", max(0, elapsed)))
        with timed(phases, "allocation_contract_validation"):
            workspace, allocation_contract, path_observations = (
                validate_allocation_contract(
                    observe_current,
                    observation.profile,
                    contract=contract,
                    path_probes=path_probes,
                )
            )
    finally:
        if isinstance(source.task, ProfileExecutable):
            profiler.executables.release_occurrence_values(source.task)
    return task_measurement(
        source.task,
        execution_provider,
        workspace,
        timing,
        phases,
        allocation_contract,
        path_observations,
    )


def warm_provider(
    invoke: Callable[[], None],
    requested_allocated_bytes: Callable[[], int],
    warmups: int,
) -> None:
    """Invoke until the provider's own allocations stop moving.

    At least ``warmups`` invocations run; after that the first pair of equal
    live-byte readings ends the warmup, and a provider still allocating after
    the stabilization budget is an error rather than a baseline.
    """

    previous = requested_allocated_bytes()
    for iteration in range(warmups + STABILIZATION_BUDGET):
        invoke()
        current = requested_allocated_bytes()
        if iteration + 1 >= warmups and current == previous:
            return
        previous = current
    raise AllocationTelemetryError(
        "provider allocations did not stabilize during task warmup"
    )


@contextmanager
def timed(phases: list[tuple[str, int]], name: str) -> Iterator[None]:
    """Append the body's wall time to the phase timings under ``name``."""

    started = time.perf_counter_ns()
    try:
        yield
    finally:
        phases.append((name, time.perf_counter_ns() - started))


def task_measurement(
    task: Callable[[], object],
    execution_provider: str,
    workspace: TaskWorkspaceProfile,
    timing: TimingObservation,
    phases: list[tuple[str, int]],
    allocation_contract: TaskAllocationContract,
    path_observations: tuple[TaskAllocationPathObservation, ...] = (),
) -> TaskMeasurement:
    """Assemble the record the profile store keeps for one task.

    Provider state is what the task-local ownership trace proves still live.
    Process-wide live-byte deltas are deliberately not an ownership signal:
    compilation artifacts, representative inputs, and stream-pending
    retirements all survive the sampling boundary.  The normalized trace has
    already excluded declared outputs and ordinary workspace, so only its
    still-live, otherwise-unbound allocations are provider state.
    """

    fixed_extents = workspace.persistent_extent_bytes
    return TaskMeasurement(
        runtime_ns=round(statistics.median(timing.samples)),
        workspace_requested_bytes=workspace.peak_requested_bytes,
        workspace_charged_bytes=workspace.peak_charged_bytes,
        workspace_extent_bytes=workspace.peak_extent_bytes,
        samples_ns=timing.samples,
        provenance=(
            f"backend-events+shadowspill-allocation-telemetry+{execution_provider}"
            + ("+bounded-retention-audit" if fixed_extents else "")
        ),
        allocation_trace=workspace.allocation_trace,
        output_input_bindings=workspace.output_input_bindings,
        persistent_extent_bytes=fixed_extents,
        representative_inputs=(
            task.representative_inputs if isinstance(task, ProfileExecutable) else ()
        ),
        phase_timings_ns=tuple(phases),
        timing_relative_mad=timing.relative_mad,
        timing_half_drift=timing.half_drift,
        timing_unstable=timing.variability > STABLE_VARIABILITY,
        allocation_contract=allocation_contract,
        allocation_path_observations=path_observations,
    )
