"""The allocation contract: probed under varied inputs, then one invariant derived."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import torch

from shadowspill.profiling.invariant import (
    AllocationPathProbe,
    derive_invariant_allocation_path,
)
from shadowspill.pytorch.capture.storage import TaskStorageContract
from shadowspill.runtime.telemetry import (
    AllocationTelemetryError,
    TaskWorkspaceProfile,
)
from shadowspill.task.allocations import (
    TaskAllocationContract,
    TaskAllocationPathObservation,
)

from ..executables import ProfileExecutable, ProfileExecutableStore
from .boundary import AllocatorBoundary
from .workspace import WorkspaceObservation

Observe = Callable[[Callable[[], object]], WorkspaceObservation]


@dataclass(frozen=True, slots=True)
class ProbedPath:
    """One workspace measurement under one representative input identity."""

    probe_index: int
    repetition: int
    allocation_contract: TaskAllocationContract
    workspace: TaskWorkspaceProfile


def probe_allocation_paths(
    boundary: AllocatorBoundary,
    stream: torch.cuda.Stream,
    executables: ProfileExecutableStore,
    executable: ProfileExecutable,
    observe: Observe,
    *,
    seeds: int,
    repetitions: int,
) -> tuple[ProfileExecutable, tuple[ProbedPath, ...]]:
    """Measure the workspace under each representative input seed, repeatedly.

    Every seed releases the previous inputs through their shared owner before
    materializing its own, so no stale wrapper keeps two identities alive.
    The executable bound to the last seed is returned with the probes.
    """

    contract = executable.artifact.storage_contract
    probes: list[ProbedPath] = []
    try:
        for probe_index in range(seeds):
            executable = executables.release_occurrence_values(executable)
            boundary.drain(stream, problem="allocation path probe")
            executable = executables.with_arguments(executable, probe_index=probe_index)
            for repetition in range(repetitions):
                measured = observe(executable).profile
                probes.append(
                    ProbedPath(
                        probe_index,
                        repetition,
                        TaskAllocationContract.capture(
                            measured.allocation_contract_trace, contract
                        ),
                        measured,
                    )
                )
    except BaseException:
        executables.release_occurrence_values(executable)
        raise
    return executable, tuple(probes)


def validate_allocation_contract(
    observe: Callable[[], WorkspaceObservation],
    baseline: TaskWorkspaceProfile,
    *,
    contract: TaskStorageContract | None,
    path_probes: Sequence[ProbedPath],
) -> tuple[
    TaskWorkspaceProfile,
    TaskAllocationContract,
    tuple[TaskAllocationPathObservation, ...],
]:
    """Derive one invariant from stable warm traces and the probe matrix.

    Two more warm measurements must repeat the baseline's contract and its
    output bindings exactly; the probes must repeat the bindings. The
    invariant is the medoid of every observed path, and the workspace
    returned is its source charged at the largest peak any path reached.
    """

    expected = TaskAllocationContract.capture(
        baseline.allocation_contract_trace, contract
    )
    warm_workspaces = [baseline]
    for repetition in range(2):
        observed = observe().profile
        warm_workspaces.append(observed)
        candidate = TaskAllocationContract.capture(
            observed.allocation_contract_trace, contract
        )
        if candidate.compatibility_digest != expected.compatibility_digest:
            raise AllocationTelemetryError(
                "task allocation contract changed across independent "
                f"profiling traces (repetition={repetition + 2}, "
                f"expected={expected.compatibility_digest}, "
                f"observed={candidate.compatibility_digest})"
            )
        if observed.output_input_bindings != baseline.output_input_bindings:
            raise AllocationTelemetryError(
                "task output/input storage bindings changed across "
                f"independent profiling traces (repetition={repetition + 2})"
            )
    for probe in path_probes:
        if probe.workspace.output_input_bindings != baseline.output_input_bindings:
            raise AllocationTelemetryError(
                "task output/input storage bindings changed across "
                "representative allocation-path probes "
                f"(probe={probe.probe_index}, repetition={probe.repetition})"
            )
    try:
        derived = derive_invariant_allocation_path(
            expected,
            tuple(
                AllocationPathProbe(
                    probe.probe_index, probe.repetition, probe.allocation_contract
                )
                for probe in path_probes
            ),
            warmed_reference_repetitions=len(warm_workspaces),
        )
    except ValueError as error:
        raise AllocationTelemetryError(
            f"task allocation paths cannot derive one invariant: {error}"
        ) from error
    candidates = (*warm_workspaces, *(probe.workspace for probe in path_probes))
    source = next(
        (
            item
            for item in candidates
            if TaskAllocationContract.capture(
                item.allocation_contract_trace, contract
            ).compatibility_digest
            == derived.source_digest
        ),
        None,
    )
    if source is None:
        raise AssertionError("the derived invariant has no source workspace")
    return (
        _charged_at_peak(source, candidates),
        derived.allocation_contract,
        derived.observations,
    )


def _charged_at_peak(
    invariant: TaskWorkspaceProfile,
    observations: Sequence[TaskWorkspaceProfile],
) -> TaskWorkspaceProfile:
    """Keep invariant ordering while charging the largest observed live-set peak."""

    peak_source = max(
        observations,
        key=lambda item: (
            item.peak_charged_bytes,
            item.peak_requested_bytes,
            len(item.allocation_contract_trace),
        ),
    )
    peak_requested = max(item.peak_requested_bytes for item in observations)
    if (
        invariant.peak_requested_bytes == peak_requested
        and invariant.peak_charged_bytes == peak_source.peak_charged_bytes
        and invariant.peak_extent_bytes == peak_source.peak_extent_bytes
    ):
        return invariant
    return replace(
        invariant,
        peak_requested_bytes=peak_requested,
        peak_charged_bytes=peak_source.peak_charged_bytes,
        peak_extent_bytes=peak_source.peak_extent_bytes,
    )
