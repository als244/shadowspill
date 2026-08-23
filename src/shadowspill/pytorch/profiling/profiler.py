"""Isolated CUDA task profiling orchestrator."""

from __future__ import annotations

import ctypes
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

import torch
from torch.utils._pytree import tree_flatten

from shadowspill.pytorch.capture.artifacts import (
    AotGraphPair,
    GraphArtifact,
    TaskInputProvenance,
    TaskInputRole,
)
from shadowspill.pytorch.compilation.compiler import CompiledTaskSet
from shadowspill.pytorch.compilation.inductor import ExecutableTaskManifest
from shadowspill.pytorch.contracts import CaptureError, ProfilingError
from shadowspill.pytorch.optimizer import (
    OpaqueOptimizerArtifact,
    materialize_opaque_optimizer,
    opaque_optimizer_outputs,
)
from shadowspill.pytorch.runtime_adapter.abi import AdapterStatistics, Allocation
from shadowspill.pytorch.runtime_adapter.failures import raise_if_allocator_failed
from shadowspill.pytorch.runtime_adapter.telemetry import (
    AllocationTelemetryError,
    TaskWorkspaceProfile,
    read_allocation_telemetry,
    start_allocation_telemetry,
    stop_allocation_telemetry,
    summarize_task_workspace,
)

from .allocation_contract import (
    TaskAllocationContract,
    TaskAllocationPathObservation,
)
from .allocation_core import AllocationPathProbe, derive_core_allocation_path
from .executables import ProfileExecutable, ProfileExecutableStore
from .records import TaskMeasurement, TaskOutputInputBinding
from .runner import ProfilableArtifact


def _timing_stability(samples: Sequence[int]) -> tuple[float, float]:
    """Return relative MAD and first/second-half median drift."""

    if not samples:
        raise ValueError("timing stability requires at least one sample")
    median = float(statistics.median(samples))
    if median <= 0:
        return (0.0, 0.0) if not any(samples) else (float("inf"), float("inf"))
    mad = float(statistics.median(abs(value - median) for value in samples)) / median
    midpoint = len(samples) // 2
    if midpoint == 0:
        return mad, 0.0
    first = float(statistics.median(samples[:midpoint]))
    second = float(statistics.median(samples[-midpoint:]))
    return mad, abs(first - second) / median


@dataclass(frozen=True, slots=True)
class _WorkspaceObservation:
    profile: Any
    execution_wall_time_ns: int
    telemetry_copy_decode_ns: int
    replay_wall_time_ns: int


@dataclass(frozen=True, slots=True)
class _TimingObservation:
    samples: tuple[int, ...]
    relative_mad: float
    half_drift: float


@dataclass(frozen=True, slots=True)
class _AllocationPathProbe:
    probe_index: int
    repetition: int
    allocation_contract: TaskAllocationContract
    output_input_bindings: tuple[TaskOutputInputBinding, ...]
    workspace: TaskWorkspaceProfile


class CudaTaskProfiler:
    """Warm and measure compiled tasks through an installed ShadowSpill slab."""

    def __init__(
        self,
        library: Any,
        *,
        device_ordinal: int,
        warmup_iterations: int = 3,
        sample_iterations: int = 5,
        telemetry_capacity: int = 1_048_576,
        allocation_probe_seeds: int = 1,
        allocation_probe_repetitions: int = 2,
    ) -> None:
        if warmup_iterations < 1:
            raise ValueError("task profiler requires at least one warmup")
        if sample_iterations < 1:
            raise ValueError("task profiler requires at least one sample")
        if telemetry_capacity < 1:
            raise ValueError("task profiler telemetry capacity must be positive")
        if allocation_probe_seeds < 1 or allocation_probe_repetitions < 2:
            raise ValueError(
                "allocation paths require at least one seed and two repetitions"
            )
        self._library = library
        self._device_ordinal = device_ordinal
        self._warmups = warmup_iterations
        self._samples = sample_iterations
        self._telemetry_capacity = telemetry_capacity
        self._allocation_probe_seeds = allocation_probe_seeds
        self._allocation_probe_repetitions = allocation_probe_repetitions
        self._next_scope_id = 1 << 62
        self._executables = ProfileExecutableStore(
            device_ordinal=device_ordinal,
            allocation_check=lambda operation: raise_if_allocator_failed(
                self._library,
                operation,
            ),
        )
        self._profiling_wall_time_ns = 0
        self._entrypoint_warmup_wall_time_ns = 0
        self._timing_events: tuple[Any, Any] | None = None
        self._device_conditioned = False
        self._last_workspace_observation: _WorkspaceObservation | None = None
        self._last_audit_timings: tuple[tuple[str, int], ...] = ()
        self._saved_control_values: dict[
            tuple[str, str | None, int], tuple[torch.Tensor | None, ...]
        ] = {}
        self._saved_control_compilation_wall_time_ns = 0

    @property
    def compilation_wall_time_ns(self) -> int:
        """Wall time spent constructing unique compiled entrypoints."""

        return self._executables.compilation_wall_time_ns

    @property
    def compilation_phase_timings_ns(self) -> tuple[tuple[str, int], ...]:
        """Non-overlapping compiler subphases aggregated across unique ABIs."""

        return self._executables.compilation_phase_timings_ns

    @property
    def compilation_phase_timings_by_contract(
        self,
    ) -> tuple[tuple[str, tuple[tuple[str, int], ...]], ...]:
        """Compiler subphases for every independently compiled structural contract."""

        return self._executables.compilation_phase_timings_by_contract

    @property
    def profiling_wall_time_ns(self) -> int:
        """Wall time spent warming, measuring, and auditing cache misses."""

        return self._profiling_wall_time_ns

    @property
    def entrypoint_warmup_wall_time_ns(self) -> int:
        """Warmup required only for entrypoints whose profile was cached."""

        return self._entrypoint_warmup_wall_time_ns

    @property
    def saved_control_compilation_wall_time_ns(self) -> int:
        """Compilation already charged to the saved-control planning phase."""

        return self._saved_control_compilation_wall_time_ns

    def measure(self, artifact: ProfilableArtifact) -> TaskMeasurement:
        """Measure one compiled graph or bounded eager optimizer task."""

        if isinstance(artifact, OpaqueOptimizerArtifact):
            profiling_started = time.perf_counter_ns()
            try:
                measurement = self._measure_opaque_optimizer(artifact)
                profiling_wall = time.perf_counter_ns() - profiling_started
                return replace(measurement, profiling_wall_time_ns=profiling_wall)
            except ProfilingError:
                raise
            except BaseException as error:
                raise _profiling_error(artifact, error) from error
            finally:
                self._profiling_wall_time_ns += (
                    time.perf_counter_ns() - profiling_started
                )
        if not isinstance(artifact, GraphArtifact):
            raise TypeError(f"unsupported profiling artifact {type(artifact).__name__}")

        digest = artifact.compatibility_digest
        executable = self._compiled(artifact)
        try:
            if not executable.example_arguments and artifact.example_arguments:
                executable = self._restore_example_arguments(executable)
            profiling_started = time.perf_counter_ns()
            measurement = self._measure_callable(
                executable,
                execution_provider=(
                    f"{executable.execution_provider}"
                    f"[fx_nodes={executable.graph_node_count}]"
                ),
            )
            measurement = replace(
                measurement,
                profiling_wall_time_ns=time.perf_counter_ns() - profiling_started,
            )
            self._profiling_wall_time_ns += measurement.profiling_wall_time_ns
            self._executables.mark_warmed(digest)
        except ProfilingError:
            self._executables.remove(digest)
            raise
        except BaseException as error:
            self._executables.remove(digest)
            raise _profiling_error(artifact, error) from error
        else:
            # The compiled function does not own its example arguments. Keeping
            # every unique contract's device examples alive until take_functions()
            # makes isolated profiling scale with the sum of model-stage
            # inputs, rather than the largest contract. Retain only the executable.
            self._executables.release_occurrence_values(executable)
            return measurement
        finally:
            del executable

    def resolve_graph_pair_controls(
        self,
        pair: AotGraphPair,
        metadata_digest: str | None = None,
    ) -> AotGraphPair:
        """Populate backward saved controls from the paired forward task."""

        compilation_before = self.compilation_wall_time_ns
        try:
            return self._resolve_graph_pair_controls(pair, metadata_digest)
        finally:
            self._saved_control_compilation_wall_time_ns += (
                self.compilation_wall_time_ns - compilation_before
            )

    def _resolve_graph_pair_controls(
        self,
        pair: AotGraphPair,
        metadata_digest: str | None,
    ) -> AotGraphPair:
        """Resolve one pair while the public wrapper accounts compilation."""

        provenance = pair.backward.input_provenance
        missing = tuple(
            position
            for position, item in enumerate(provenance[: pair.saved_value_count])
            if item.role is TaskInputRole.CONTROL and item.representative_value is None
        )
        if not missing:
            return pair
        key = (
            pair.forward.compatibility_digest,
            metadata_digest,
            pair.saved_value_count,
        )
        values = self._saved_control_values.get(key)
        if values is None:
            values = self._execute_saved_control_producer(pair)
            self._saved_control_values[key] = values
        rebound = tuple(
            _bind_saved_control(item, values[position]) if position in missing else item
            for position, item in enumerate(provenance)
        )
        backward = pair.backward.rebind_examples(
            pair.backward.example_arguments,
            input_provenance=rebound,
        )
        return replace(pair, backward=backward)

    def _execute_saved_control_producer(
        self,
        pair: AotGraphPair,
    ) -> tuple[torch.Tensor | None, ...]:
        """Run one forward task and snapshot only non-floating saved leaves."""

        executable = self._compiled(pair.forward)
        if not executable.example_arguments:
            executable = self._restore_example_arguments(executable)
        torch.cuda.set_device(self._device_ordinal)
        stream = torch.cuda.current_stream(self._device_ordinal)
        scope_id = self._open_allocation_scope()
        scope_open = True
        output: object | None = None
        try:
            output = executable()
            leaves, _ = tree_flatten(output)
            original_count = pair.forward.output_count - pair.saved_value_count
            saved = leaves[original_count:]
            if len(saved) != pair.saved_value_count:
                raise CaptureError("paired forward changed its saved-value arity")
            values = tuple(
                _snapshot_saved_control(value)
                if pair.backward.input_provenance[position].role
                is TaskInputRole.CONTROL
                else None
                for position, value in enumerate(saved)
            )
            output = None
            self._close_allocation_scope(scope_id, stream)
            scope_open = False
            stream.synchronize()
            self._diagnose_allocator_idle(problem="saved-control producer")
            return values
        except BaseException:
            if scope_open:
                self._library.shadowspill_pytorch_allocation_scope_abort()
            raise
        finally:
            output = None
            self._executables.release_occurrence_values(executable)

    def prepare_manifests(
        self,
        artifacts: Sequence[ProfilableArtifact],
        *,
        progress: Callable[[int, int, str, str], None] | None = None,
    ) -> dict[str, ExecutableTaskManifest]:
        """Compile only missing storage manifests without retaining examples."""

        return self._executables.prepare_manifests(artifacts, progress=progress)

    def _measure_callable(
        self,
        executable: Callable[[], object],
        *,
        execution_provider: str = "bounded-eager",
    ) -> TaskMeasurement:
        """Measure a warmed no-argument task through the allocator boundary."""

        owned_occurrence = (
            executable if isinstance(executable, ProfileExecutable) else None
        )
        torch.cuda.set_device(self._device_ordinal)
        stream = torch.cuda.current_stream(self._device_ordinal)
        phase_timings: list[tuple[str, int]] = []
        try:
            self._condition_profile_device(stream, phase_timings)
            persistent_baseline = self._requested_allocated_bytes()
            # Provider compilation, autotuning, and shape-keyed initialization
            # are planning-time setup, not alternative task allocation paths.
            # Warm them before varying representative input identities.
            persistent_high_water = self._warm_profile_provider(
                executable,
                stream,
                baseline=persistent_baseline,
                phase_timings=phase_timings,
            )
            path_probes: tuple[_AllocationPathProbe, ...] = ()
            if isinstance(executable, ProfileExecutable):
                # Release A through its shared owner before materializing B.
                # Even the caller's stale wrapper now observes an empty owner.
                executable = self._executables.release_occurrence_values(executable)
                stream.synchronize()
                self._diagnose_allocator_idle(
                    problem="pre-probe representative input release"
                )
                executable, path_probes = self._probe_allocation_paths(
                    executable,
                    stream,
                    phase_timings,
                )
                owned_occurrence = executable
                stabilization_started = time.perf_counter_ns()
                persistent_high_water = max(
                    persistent_high_water,
                    self._requested_allocated_bytes(),
                )
                persistent_high_water = self._await_stable_provider_state(
                    executable,
                    stream,
                    persistent_high_water,
                )
                phase_timings.append(
                    (
                        "post_probe_stabilization",
                        time.perf_counter_ns() - stabilization_started,
                    )
                )
            timing = self._collect_timing_samples(executable, stream, phase_timings)
            self._library.shadowspill_pytorch_allocator_wait_idle()
            persistent_high_water = max(
                persistent_high_water, self._requested_allocated_bytes()
            )
            (
                workspace,
                persistent_high_water,
                allocation_contract,
                path_observations,
            ) = self._profile_workspace(
                executable,
                stream,
                persistent_high_water=persistent_high_water,
                path_probes=path_probes,
                phase_timings=phase_timings,
            )
            fixed_extents = _persistent_profile_extents(workspace)
            return _task_measurement(
                executable,
                execution_provider,
                workspace,
                timing,
                fixed_extents,
                phase_timings,
                allocation_contract,
                path_observations,
            )
        finally:
            if owned_occurrence is not None:
                self._executables.release_occurrence_values(owned_occurrence)

    def _probe_allocation_paths(
        self,
        executable: ProfileExecutable,
        stream: torch.cuda.Stream,
        phase_timings: list[tuple[str, int]],
    ) -> tuple[ProfileExecutable, tuple[_AllocationPathProbe, ...]]:
        """Probe value/identity allocation paths after provider warmup."""

        started = time.perf_counter_ns()
        contract = executable.artifact.storage_contract
        observations: list[_AllocationPathProbe] = []
        try:
            for probe_index in range(self._allocation_probe_seeds):
                executable = self._executables.release_occurrence_values(executable)
                stream.synchronize()
                self._library.shadowspill_pytorch_allocator_wait_idle()
                executable = self._executables.with_arguments(
                    executable,
                    probe_index=probe_index,
                )
                for repetition in range(self._allocation_probe_repetitions):
                    measured = self._measure_workspace(executable, stream)
                    observations.append(
                        _AllocationPathProbe(
                            probe_index,
                            repetition,
                            TaskAllocationContract.capture(
                                measured.allocation_contract_trace,
                                contract,
                            ),
                            measured.output_input_bindings,
                            measured,
                        )
                    )
        except BaseException:
            self._executables.release_occurrence_values(executable)
            raise
        phase_timings.append(
            ("allocation_path_probes", time.perf_counter_ns() - started)
        )
        return executable, tuple(observations)

    def _condition_profile_device(
        self,
        stream: torch.cuda.Stream,
        phase_timings: list[tuple[str, int]],
    ) -> None:
        if self._device_conditioned:
            return
        started = time.perf_counter_ns()
        self._condition_device(stream)
        phase_timings.append(("device_conditioning", time.perf_counter_ns() - started))

    def _warm_profile_provider(
        self,
        executable: Callable[[], object],
        stream: torch.cuda.Stream,
        *,
        baseline: int,
        phase_timings: list[tuple[str, int]],
    ) -> int:
        started = time.perf_counter_ns()
        for _ in range(self._warmups):
            self._invoke_profile_task(executable, stream)
        stream.synchronize()
        self._library.shadowspill_pytorch_allocator_wait_idle()
        high_water = max(baseline, self._requested_allocated_bytes())
        high_water = self._await_stable_provider_state(
            executable,
            stream,
            high_water,
        )
        phase_timings.append(("provider_warmup", time.perf_counter_ns() - started))
        return high_water

    def _await_stable_provider_state(
        self,
        executable: Callable[[], object],
        stream: torch.cuda.Stream,
        high_water: int,
    ) -> int:
        previous = self._requested_allocated_bytes()
        for _ in range(16):
            self._invoke_profile_task(executable, stream)
            current = self._requested_allocated_bytes()
            high_water = max(high_water, current)
            if current == previous:
                return high_water
            previous = current
        raise AllocationTelemetryError(
            "provider allocations did not stabilize during task warmup"
        )

    def _collect_timing_samples(
        self,
        executable: Callable[[], object],
        stream: torch.cuda.Stream,
        phase_timings: list[tuple[str, int]],
    ) -> _TimingObservation:
        started = time.perf_counter_ns()
        samples: list[int] = []
        while True:
            target = self._samples if not samples else min(15, len(samples) + 2)
            samples.extend(
                self._measure_task_once(executable, stream)
                for _ in range(target - len(samples))
            )
            relative_mad, half_drift = _timing_stability(samples)
            if (
                self._samples < 5
                or max(relative_mad, half_drift) <= 0.03
                or len(samples) >= 15
            ):
                break
        phase_timings.append(("timing_samples", time.perf_counter_ns() - started))
        return _TimingObservation(tuple(samples), relative_mad, half_drift)

    def _profile_workspace(
        self,
        executable: Callable[[], object],
        stream: torch.cuda.Stream,
        *,
        persistent_high_water: int,
        path_probes: tuple[_AllocationPathProbe, ...],
        phase_timings: list[tuple[str, int]],
    ) -> tuple[
        TaskWorkspaceProfile,
        int,
        TaskAllocationContract,
        tuple[TaskAllocationPathObservation, ...],
    ]:
        started = time.perf_counter_ns()
        workspace, high_water = self._audit_workspace_retention(
            executable,
            stream,
            persistent_high_water=persistent_high_water,
        )
        workspace_timings = self._last_audit_timings
        phase_timings.extend(workspace_timings)
        accounted = sum(duration for _name, duration in workspace_timings)
        phase_timings.append(
            ("retention_audit", max(0, time.perf_counter_ns() - started - accounted))
        )
        contract_started = time.perf_counter_ns()
        workspace, allocation_contract, path_observations = (
            self._validate_allocation_contract(
                executable,
                stream,
                workspace,
                path_probes=path_probes,
            )
        )
        phase_timings.append(
            (
                "allocation_contract_validation",
                time.perf_counter_ns() - contract_started,
            )
        )
        return workspace, high_water, allocation_contract, path_observations

    def _validate_allocation_contract(
        self,
        executable: Callable[[], object],
        stream: torch.cuda.Stream,
        baseline: TaskWorkspaceProfile,
        *,
        path_probes: tuple[_AllocationPathProbe, ...],
    ) -> tuple[
        TaskWorkspaceProfile,
        TaskAllocationContract,
        tuple[TaskAllocationPathObservation, ...],
    ]:
        """Derive one core from stable warm traces and the probe matrix."""

        contract = (
            executable.artifact.storage_contract
            if isinstance(executable, ProfileExecutable)
            else None
        )
        expected = TaskAllocationContract.capture(
            baseline.allocation_contract_trace, contract
        )
        warm_workspaces = [baseline]
        for repetition in range(2):
            observed = self._measure_workspace(executable, stream)
            warm_workspaces.append(observed)
            candidate = TaskAllocationContract.capture(
                observed.allocation_contract_trace, contract
            )
            if candidate.compatibility_digest != expected.compatibility_digest:
                raise AllocationTelemetryError(
                    "compiled task allocation contract changed across independent "
                    f"profiling traces (repetition={repetition + 2}, "
                    f"expected={expected.compatibility_digest}, "
                    f"observed={candidate.compatibility_digest})"
                )
            if observed.output_input_bindings != baseline.output_input_bindings:
                raise AllocationTelemetryError(
                    "compiled task output/input storage bindings changed across "
                    f"independent profiling traces (repetition={repetition + 2})"
                )
        for probe in path_probes:
            if probe.output_input_bindings != baseline.output_input_bindings:
                raise AllocationTelemetryError(
                    "compiled task output/input storage bindings changed across "
                    "representative allocation-path probes "
                    f"(probe={probe.probe_index}, repetition={probe.repetition})"
                )
        try:
            derived = derive_core_allocation_path(
                expected,
                tuple(
                    AllocationPathProbe(
                        probe.probe_index,
                        probe.repetition,
                        probe.allocation_contract,
                    )
                    for probe in path_probes
                ),
                warmed_reference_repetitions=len(warm_workspaces),
            )
        except ValueError as error:
            raise AllocationTelemetryError(
                f"compiled task allocation paths cannot derive one fixed core: {error}"
            ) from error
        candidates = (*warm_workspaces, *(item.workspace for item in path_probes))
        core_workspace = next(
            (
                item
                for item in candidates
                if TaskAllocationContract.capture(
                    item.allocation_contract_trace,
                    contract,
                ).compatibility_digest
                == derived.source_digest
            ),
            None,
        )
        if core_workspace is None:
            raise AssertionError("derived allocation core has no source workspace")
        return (
            _conservative_core_workspace(core_workspace, candidates),
            derived.allocation_contract,
            derived.observations,
        )

    def _invoke_profile_task(
        self,
        executable: Callable[[], object],
        stream: torch.cuda.Stream,
    ) -> None:
        scope_id = self._open_allocation_scope()
        try:
            output = executable()
            del output
            self._close_allocation_scope(scope_id, stream)
        except BaseException:
            self._library.shadowspill_pytorch_allocation_scope_abort()
            raise
        # Drain before the next invocation. Retirement is asynchronous, so
        # without this a warmup loop runs every iteration while the previous
        # ones still hold their ranges - measured at 4,271 leases and 7.4 GiB
        # outstanding, against a pool that has to fit the task being measured.
        stream.synchronize()
        self._library.shadowspill_pytorch_allocator_wait_idle()

    def _measure_task_once(
        self,
        executable: Callable[[], object],
        stream: torch.cuda.Stream,
    ) -> int:
        start, finish = self._timing_event_pair()
        scope_id = self._open_allocation_scope()
        try:
            start.record(stream)
            output = executable()
            del output
            finish.record(stream)
            self._close_allocation_scope(scope_id, stream)
        except BaseException:
            self._library.shadowspill_pytorch_allocation_scope_abort()
            raise
        finish.synchronize()
        self._library.shadowspill_pytorch_allocator_wait_idle()
        elapsed_ms = cast(float, start.elapsed_time(finish))
        return max(0, round(elapsed_ms * 1_000_000))

    def _timing_event_pair(self) -> tuple[Any, Any]:
        if self._timing_events is None:
            event_factory: Any = torch.cuda.Event
            self._timing_events = (
                event_factory(enable_timing=True),
                event_factory(enable_timing=True),
            )
        return self._timing_events

    def _condition_device(self, stream: torch.cuda.Stream) -> None:
        """Warm clocks/provider state once using bounded preallocated GEMM."""

        if self._device_conditioned:
            return
        shape = (2048, 2048)
        device = torch.device("cuda", self._device_ordinal)
        left = torch.randn(shape, dtype=torch.bfloat16, device=device)
        right = torch.randn(shape, dtype=torch.bfloat16, device=device)
        output = torch.empty(shape, dtype=torch.bfloat16, device=device)
        samples: list[int] = []
        start, finish = self._timing_event_pair()
        for _ in range(64):
            start.record(stream)
            torch.mm(left, right, out=output)
            finish.record(stream)
            finish.synchronize()
            samples.append(max(1, round(start.elapsed_time(finish) * 1_000_000)))
            if len(samples) >= 3:
                recent = samples[-3:]
                median = float(statistics.median(recent))
                if median > 0 and (max(recent) - min(recent)) / median <= 0.02:
                    break
        del output
        del right
        del left
        stream.synchronize()
        self._library.shadowspill_pytorch_allocator_wait_idle()
        self._device_conditioned = True

    def _open_allocation_scope(self) -> int:
        scope_id = self._next_scope_id
        self._next_scope_id += 1
        status = int(
            self._library.shadowspill_pytorch_allocation_scope_begin(scope_id)
        )
        if status != 0:
            raise CaptureError(
                f"profiling allocation scope begin failed with status {status}"
            )
        return scope_id

    def _close_allocation_scope(
        self, scope_id: int, stream: torch.cuda.Stream
    ) -> None:
        status = int(
            self._library.shadowspill_pytorch_allocation_scope_end(
                scope_id, stream.cuda_stream
            )
        )
        if status != 0:
            raise CaptureError(
                f"profiling allocation scope end failed with status {status}"
            )

    def _audit_workspace_retention(
        self,
        executable: Callable[[], object],
        stream: torch.cuda.Stream,
        *,
        persistent_high_water: int,
        maximum_iterations: int = 16,
    ) -> tuple[Any, int]:
        """Distinguish bounded provider caches from unbounded task leakage."""

        previous = self._requested_allocated_bytes()
        stable_observations = 0
        workspace: Any | None = None
        execution_ns = 0
        copy_decode_ns = 0
        replay_ns = 0
        self._last_audit_timings = ()
        for _ in range(maximum_iterations):
            self._last_workspace_observation = None
            workspace = self._measure_workspace(executable, stream)
            observation = self._last_workspace_observation
            if observation is not None:
                execution_ns += observation.execution_wall_time_ns
                copy_decode_ns += observation.telemetry_copy_decode_ns
                replay_ns += observation.replay_wall_time_ns
            current = self._requested_allocated_bytes()
            persistent_high_water = max(persistent_high_water, current)
            if not workspace.persistent_extent_bytes:
                self._last_audit_timings = (
                    ("workspace_execution", execution_ns),
                    ("telemetry_copy_decode", copy_decode_ns),
                    ("workspace_replay", replay_ns),
                )
                return workspace, persistent_high_water
            if current == previous:
                stable_observations += 1
            else:
                stable_observations = 0
            if stable_observations >= 1:
                self._last_audit_timings = (
                    ("workspace_execution", execution_ns),
                    ("telemetry_copy_decode", copy_decode_ns),
                    ("workspace_replay", replay_ns),
                )
                return workspace, persistent_high_water
            previous = current
        if workspace is None:
            raise AssertionError("workspace retention audit did not execute")
        raise AllocationTelemetryError(
            "task retains anonymous allocations without reaching a bounded "
            f"live-byte baseline after {maximum_iterations} invocations; "
            f"latest={workspace.persistent_extent_bytes}"
        )

    def _requested_allocated_bytes(self) -> int:
        return int(self._allocator_statistics().runtime.requested_allocated_bytes)

    def _allocator_statistics(self) -> AdapterStatistics:
        statistics = AdapterStatistics()
        status = int(
            self._library.shadowspill_pytorch_allocator_statistics(
                ctypes.byref(statistics)
            )
        )
        if status != 0:
            raise AllocationTelemetryError(
                f"allocator statistics failed during profiling with status {status}"
            )
        return statistics

    def _diagnose_allocator_idle(
        self,
        *,
        problem: str,
    ) -> None:
        """Block on the runtime's progress-safe quiescence boundary."""

        status = int(self._library.shadowspill_pytorch_allocator_wait_idle())
        if status == 0:
            return
        detail = f"status={status}"
        if hasattr(self._library, "shadowspill_pytorch_allocator_statistics"):
            statistics = self._allocator_statistics().runtime
            detail = (
                f"{detail} pending={statistics.pending_retirements} "
                f"fenced={statistics.retirement_records_fenced} "
                f"evented={statistics.retirement_records_evented} "
                f"preparing={statistics.retirement_records_preparing} "
                f"unfenced={statistics.retirement_records_unfenced} "
                f"actions={statistics.queued_actions}"
            )
        raise AllocationTelemetryError(
            f"allocator failed to become idle during {problem}: {detail}"
        )

    def _measure_opaque_optimizer(
        self, artifact: OpaqueOptimizerArtifact
    ) -> TaskMeasurement:
        if artifact.profile_output_names:
            return self._measure_initial_opaque_optimizer(artifact)
        optimizer = materialize_opaque_optimizer(
            artifact, device_ordinal=self._device_ordinal
        )

        def update(
            profiled_optimizer: torch.optim.Optimizer = optimizer,
        ) -> object:
            with torch.no_grad():
                return profiled_optimizer.step()

        measurement = self._measure_callable(
            update, execution_provider="opaque-optimizer"
        )
        del update
        del optimizer
        return measurement

    def _measure_initial_opaque_optimizer(
        self,
        artifact: OpaqueOptimizerArtifact,
    ) -> TaskMeasurement:
        """Profile a state-creating first step from an independent baseline.

        Reusing one optimizer would turn every invocation after the first into
        the recurrent update and silently omit lazy-state allocations.  Each
        sample therefore materializes the same storage-free optimizer before
        opening the measured task boundary, then destroys it after all output
        allocations have been classified.
        """

        torch.cuda.set_device(self._device_ordinal)
        stream = torch.cuda.current_stream(self._device_ordinal)
        phase_timings: list[tuple[str, int]] = []
        self._condition_profile_device(stream, phase_timings)
        persistent_baseline = self._requested_allocated_bytes()

        warmup_started = time.perf_counter_ns()
        persistent_high_water = persistent_baseline
        previous = persistent_baseline
        for iteration in range(self._warmups + 16):
            self._with_initial_opaque_callable(
                artifact,
                stream,
                lambda executable: self._invoke_profile_task(executable, stream),
            )
            current = self._requested_allocated_bytes()
            persistent_high_water = max(persistent_high_water, current)
            if iteration + 1 >= self._warmups and current == previous:
                break
            previous = current
        else:
            raise AllocationTelemetryError(
                "opaque optimizer provider allocations did not stabilize "
                "during first-step warmup"
            )
        phase_timings.append(
            ("provider_warmup", time.perf_counter_ns() - warmup_started)
        )

        timing_started = time.perf_counter_ns()
        samples: list[int] = []
        while True:
            target = self._samples if not samples else min(15, len(samples) + 2)
            samples.extend(
                self._with_initial_opaque_callable(
                    artifact,
                    stream,
                    lambda executable: self._measure_task_once(executable, stream),
                )
                for _ in range(target - len(samples))
            )
            relative_mad, half_drift = _timing_stability(samples)
            if (
                self._samples < 5
                or max(relative_mad, half_drift) <= 0.03
                or len(samples) >= 15
            ):
                break
        timing = _TimingObservation(tuple(samples), relative_mad, half_drift)
        phase_timings.append(
            ("timing_samples", time.perf_counter_ns() - timing_started)
        )

        workspace_started = time.perf_counter_ns()
        workspace = self._with_initial_opaque_callable(
            artifact,
            stream,
            lambda executable: self._measure_workspace(executable, stream),
        )
        observation = self._last_workspace_observation
        if observation is not None:
            phase_timings.extend(
                (
                    ("workspace_execution", observation.execution_wall_time_ns),
                    ("telemetry_copy_decode", observation.telemetry_copy_decode_ns),
                    ("workspace_replay", observation.replay_wall_time_ns),
                )
            )
            accounted = (
                observation.execution_wall_time_ns
                + observation.telemetry_copy_decode_ns
                + observation.replay_wall_time_ns
            )
        else:
            accounted = 0
        phase_timings.append(
            (
                "retention_audit",
                max(0, time.perf_counter_ns() - workspace_started - accounted),
            )
        )

        contract_started = time.perf_counter_ns()
        allocation_contract = TaskAllocationContract.capture(
            workspace.allocation_contract_trace
        )
        for repetition in range(2):
            observed = self._with_initial_opaque_callable(
                artifact,
                stream,
                lambda executable: self._measure_workspace(executable, stream),
            )
            candidate = TaskAllocationContract.capture(
                observed.allocation_contract_trace
            )
            if (
                candidate.compatibility_digest
                != allocation_contract.compatibility_digest
            ):
                raise AllocationTelemetryError(
                    "opaque optimizer first-step allocation contract changed across "
                    f"independent profiles (repetition={repetition + 2}, "
                    f"expected={allocation_contract.compatibility_digest}, "
                    f"observed={candidate.compatibility_digest})"
                )
            if observed.output_input_bindings != workspace.output_input_bindings:
                raise AllocationTelemetryError(
                    "opaque optimizer first-step output bindings changed across "
                    f"independent profiles (repetition={repetition + 2})"
                )
        phase_timings.append(
            (
                "allocation_contract_validation",
                time.perf_counter_ns() - contract_started,
            )
        )
        persistent_high_water = max(
            persistent_high_water,
            self._requested_allocated_bytes(),
        )
        fixed_extents = _persistent_profile_extents(workspace)

        def measured_callable() -> object:
            raise AssertionError("first-step profile callable is not retained")

        return _task_measurement(
            measured_callable,
            "opaque-optimizer-initial",
            workspace,
            timing,
            fixed_extents,
            phase_timings,
            allocation_contract,
        )

    def _with_initial_opaque_callable(
        self,
        artifact: OpaqueOptimizerArtifact,
        stream: torch.cuda.Stream,
        operation: Callable[[Callable[[], object]], Any],
    ) -> Any:
        optimizer = materialize_opaque_optimizer(
            artifact,
            device_ordinal=self._device_ordinal,
        )

        def update() -> object:
            with torch.no_grad():
                optimizer.step()
            return tuple(
                binding.tensor
                for binding in opaque_optimizer_outputs(
                    artifact,
                    optimizer,
                    device_ordinal=self._device_ordinal,
                )
            )

        try:
            return operation(update)
        finally:
            del update
            stream.synchronize()
            self._diagnose_allocator_idle(problem="opaque optimizer first step")

    def take_compiled_tasks(
        self,
        artifacts: Sequence[ProfilableArtifact],
        *,
        progress: Callable[[int, int, str, str], None] | None = None,
    ) -> CompiledTaskSet:
        """Transfer warmed entrypoints and their optimized storage contracts."""

        return self._executables.take_selected(
            artifacts,
            warmup=self._warm_selected_entrypoint,
            progress=progress,
        )

    def _warm_selected_entrypoint(
        self,
        executable: ProfileExecutable,
        digest: str,
    ) -> None:
        stream = torch.cuda.current_stream(self._device_ordinal)
        started = time.perf_counter_ns()
        try:
            for _ in range(self._warmups):
                self._invoke_profile_task(executable, stream)
            self._diagnose_allocator_idle(problem=f"compiled entrypoint {digest}")
        except ProfilingError:
            raise
        except BaseException as error:
            raise _profiling_error(executable.artifact, error) from error
        finally:
            self._entrypoint_warmup_wall_time_ns += time.perf_counter_ns() - started

    def discard_compiled_tasks(self) -> None:
        """Drop compiled but unselected callables retained during planning."""

        self._executables.discard()

    def _compiled(self, artifact: GraphArtifact) -> ProfileExecutable:
        return self._executables.get(artifact)

    def _restore_example_arguments(
        self, executable: ProfileExecutable
    ) -> ProfileExecutable:
        return self._executables.with_arguments(executable)

    def _measure_workspace(
        self, executable: Callable[[], object], stream: torch.cuda.Stream
    ) -> Any:
        """Measure one task, refusing a measurement the record cannot describe.

        The allocation contract is derived from the recorded events, so a
        truncated record does not describe the task that ran - it describes a
        prefix of it. Re-running the task is not a way out: its scope has
        closed and its outputs have been released. So the only honest
        outcomes are a complete record or an error naming the limit.
        """

        profile = self._measure_workspace_once(executable, stream)
        if self._allocation_events_overflowed():
            raise CaptureError(
                "allocation telemetry overflowed at "
                f"{self._telemetry_capacity} events, so the recorded "
                "allocations describe only part of this task; raise "
                "telemetry_capacity to profile it"
            )
        return profile

    def _allocation_events_overflowed(self) -> bool:
        """Whether the last measurement filled the event record."""

        return bool(
            self._allocator_statistics().runtime.allocation_event_overflow
        )

    def _measure_workspace_once(
        self, executable: Callable[[], object], stream: torch.cuda.Stream
    ) -> Any:
        task_id = self._next_scope_id
        self._next_scope_id += 1
        execution_started = time.perf_counter_ns()
        start_allocation_telemetry(self._library, capacity=self._telemetry_capacity)
        task_open = False
        output: object | None = None
        primary_error: BaseException | None = None
        try:
            status = int(
                self._library.shadowspill_pytorch_allocation_scope_begin(task_id)
            )
            if status != 0:
                raise CaptureError(
                    "profiling allocation scope begin failed with status "
                    f"{status}"
                )
            task_open = True
            output = executable()
            output_allocations, output_input_bindings = self._output_allocation_views(
                output,
                inputs=(
                    executable.example_arguments
                    if isinstance(executable, ProfileExecutable)
                    else ()
                ),
            )
            # Profiling does not retain task results. Release them while the
            # task range is still active so output-dependent temporary frees
            # remain attributable to this contract. The allocator retires their
            # physical ranges against the active compute stream.
            output = None
            status = int(
                self._library.shadowspill_pytorch_allocation_scope_end(
                    task_id, stream.cuda_stream
                )
            )
            task_open = False
            if status != 0:
                raise CaptureError(
                    "profiling allocation scope end failed with status "
                    f"{status}"
                )
            stream.synchronize()
            self._library.shadowspill_pytorch_allocator_wait_idle()
        except BaseException as error:
            primary_error = error
            if task_open:
                self._library.shadowspill_pytorch_allocation_scope_abort()
            raise
        finally:
            try:
                stop_allocation_telemetry(self._library)
            except BaseException as cleanup_error:
                if primary_error is None:
                    raise
                primary_error.add_note(
                    f"allocation telemetry cleanup also failed: {cleanup_error}"
                )
        execution_wall = time.perf_counter_ns() - execution_started
        copy_started = time.perf_counter_ns()
        events = read_allocation_telemetry(self._library)
        copy_wall = time.perf_counter_ns() - copy_started
        replay_started = time.perf_counter_ns()
        profile = summarize_task_workspace(
            events,
            task_id=task_id,
            output_allocation_views=output_allocations,
            output_input_bindings=output_input_bindings,
        )
        replay_wall = time.perf_counter_ns() - replay_started
        self._last_workspace_observation = _WorkspaceObservation(
            profile,
            execution_wall,
            copy_wall,
            replay_wall,
        )
        return profile

    def _output_allocation_views(
        self,
        output: object,
        *,
        inputs: Sequence[object] = (),
    ) -> tuple[
        dict[int, tuple[tuple[int, int], ...]],
        tuple[TaskOutputInputBinding, ...],
    ]:
        views_by_allocation: dict[int, list[tuple[int, int]]] = {}
        input_by_allocation: dict[int, int] = {}
        for input_position, value in enumerate(inputs):
            if not isinstance(value, torch.Tensor) or not value.is_cuda:
                continue
            address = value.untyped_storage().data_ptr()
            if address == 0:
                continue
            allocation = self._allocation_for_pointer(address)
            input_by_allocation.setdefault(
                int(allocation.allocation_id), input_position
            )
        input_bindings: list[TaskOutputInputBinding] = []
        leaves, _ = tree_flatten(output)
        for leaf_index, leaf in enumerate(leaves):
            if not isinstance(leaf, torch.Tensor) or not leaf.is_cuda:
                continue
            address = leaf.untyped_storage().data_ptr()
            if address == 0:
                continue
            allocation = self._allocation_for_pointer(address)
            allocation_pointer = int(allocation.pointer or 0)
            view_pointer = int(leaf.data_ptr())
            offset_bytes = view_pointer - allocation_pointer
            if offset_bytes < 0 or offset_bytes > int(allocation.requested_bytes):
                raise CaptureError(
                    "compiled output view lies outside its allocator record"
                )
            donated_input_position = input_by_allocation.get(
                int(allocation.allocation_id)
            )
            if donated_input_position is not None:
                input_bindings.append(
                    TaskOutputInputBinding(
                        leaf_index,
                        donated_input_position,
                        offset_bytes,
                    )
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

    def _allocation_for_pointer(self, address: int) -> Allocation:
        allocation = Allocation()
        status = int(
            self._library.shadowspill_pytorch_allocation_for_pointer(
                address, ctypes.byref(allocation)
            )
        )
        if status != 0:
            raise CaptureError(
                "compiled task returned storage outside the ShadowSpill slab"
            )
        return allocation


def _profiling_error(
    artifact: GraphArtifact | OpaqueOptimizerArtifact,
    cause: BaseException,
) -> ProfilingError:
    kind: str
    if isinstance(artifact, GraphArtifact):
        kind = artifact.kind
        operators = tuple(artifact.operator_targets)
    else:
        kind = "opaque_optimizer"
        operators = ()
    operator_text = ", ".join(operators) or "none"
    return ProfilingError(
        "ShadowSpill failed to profile structural contract "
        f"{artifact.compatibility_digest} "
        f"(kind={kind}, operators=[{operator_text}]): {cause}",
        structural_contract=artifact.compatibility_digest,
        task_kind=kind,
        operators=operators,
    )


def _bind_saved_control(
    provenance: TaskInputProvenance,
    value: torch.Tensor | None,
) -> TaskInputProvenance:
    if value is None:
        raise CaptureError(
            "paired forward did not produce an authentic saved control: "
            f"source={provenance.source}"
        )
    return replace(provenance, representative_value=value)


def _snapshot_saved_control(value: object) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise CaptureError("paired forward saved control is not a tensor")
    if value.is_floating_point() or value.is_complex():
        raise CaptureError("paired forward saved control has a continuous dtype")
    source = value.detach().to(device="cpu")
    result = torch.empty_strided(
        tuple(source.shape),
        tuple(source.stride()),
        dtype=source.dtype,
        device="cpu",
    )
    result.copy_(source)
    return result


def _persistent_profile_extents(
    workspace: TaskWorkspaceProfile,
) -> tuple[int, ...]:
    """Return provider state proven by the task-local ownership trace.

    Process-wide live-byte deltas are deliberately not an ownership signal:
    compilation artifacts, representative inputs, and stream-pending
    retirements can all survive across the sampling boundary.  The normalized
    task trace has already excluded declared outputs and ordinary workspace;
    only its still-live, otherwise-unbound allocations are provider state.
    """

    return workspace.persistent_extent_bytes


def _conservative_core_workspace(
    core: TaskWorkspaceProfile,
    observations: Sequence[TaskWorkspaceProfile],
) -> TaskWorkspaceProfile:
    """Keep core ordering while charging the largest observed live-set peak."""

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
        core.peak_requested_bytes == peak_requested
        and core.peak_charged_bytes == peak_source.peak_charged_bytes
        and core.peak_extent_bytes == peak_source.peak_extent_bytes
    ):
        return core
    return replace(
        core,
        peak_requested_bytes=peak_requested,
        peak_charged_bytes=peak_source.peak_charged_bytes,
        peak_extent_bytes=peak_source.peak_extent_bytes,
    )


def _task_measurement(
    executable: Callable[[], object],
    execution_provider: str,
    workspace: TaskWorkspaceProfile,
    timing: _TimingObservation,
    fixed_extents: tuple[int, ...],
    phase_timings: list[tuple[str, int]],
    allocation_contract: TaskAllocationContract,
    allocation_path_observations: tuple[TaskAllocationPathObservation, ...] = (),
) -> TaskMeasurement:
    variability = max(timing.relative_mad, timing.half_drift)
    return TaskMeasurement(
        runtime_ns=round(statistics.median(timing.samples)),
        workspace_requested_bytes=workspace.peak_requested_bytes,
        workspace_charged_bytes=workspace.peak_charged_bytes,
        workspace_extent_bytes=workspace.peak_extent_bytes,
        samples_ns=timing.samples,
        provenance=(
            f"cuda-events+shadowspill-allocation-telemetry+{execution_provider}"
            + ("+bounded-retention-audit" if fixed_extents else "")
        ),
        allocation_trace=workspace.allocation_trace,
        output_input_bindings=workspace.output_input_bindings,
        persistent_extent_bytes=fixed_extents,
        representative_inputs=(
            executable.representative_inputs
            if isinstance(executable, ProfileExecutable)
            else ()
        ),
        phase_timings_ns=tuple(phase_timings),
        timing_relative_mad=timing.relative_mad,
        timing_half_drift=timing.half_drift,
        timing_unstable=variability > 0.03,
        allocation_contract=allocation_contract,
        allocation_path_observations=allocation_path_observations,
    )
