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

from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.compilation.compiler import CompiledTaskSet
from shadowspill.pytorch.compilation.inductor import ExecutableTaskManifest
from shadowspill.pytorch.contracts import CaptureError, ProfilingError
from shadowspill.pytorch.optimizer import (
    OpaqueOptimizerArtifact,
    materialize_opaque_optimizer,
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

from .allocation_abi import TaskAllocationABI
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


class CudaTaskProfiler:
    """Warm and measure compiled tasks through an installed ShadowSpill slab."""

    def __init__(
        self,
        library: Any,
        *,
        device_ordinal: int,
        warmup_iterations: int = 3,
        sample_iterations: int = 5,
        telemetry_capacity: int = 65_536,
    ) -> None:
        if warmup_iterations < 1:
            raise ValueError("task profiler requires at least one warmup")
        if sample_iterations < 1:
            raise ValueError("task profiler requires at least one sample")
        if telemetry_capacity < 1:
            raise ValueError("task profiler telemetry capacity must be positive")
        self._library = library
        self._device_ordinal = device_ordinal
        self._warmups = warmup_iterations
        self._samples = sample_iterations
        self._telemetry_capacity = telemetry_capacity
        self._next_task_id = 1 << 62
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

    @property
    def compilation_wall_time_ns(self) -> int:
        """Wall time spent constructing unique compiled entrypoints."""

        return self._executables.compilation_wall_time_ns

    @property
    def compilation_phase_timings_ns(self) -> tuple[tuple[str, int], ...]:
        """Non-overlapping compiler subphases aggregated across unique ABIs."""

        return self._executables.compilation_phase_timings_ns

    @property
    def compilation_phase_timings_by_abi(
        self,
    ) -> tuple[tuple[str, tuple[tuple[str, int], ...]], ...]:
        """Compiler subphases for every independently compiled structural ABI."""

        return self._executables.compilation_phase_timings_by_abi

    @property
    def profiling_wall_time_ns(self) -> int:
        """Wall time spent warming, measuring, and auditing cache misses."""

        return self._profiling_wall_time_ns

    @property
    def entrypoint_warmup_wall_time_ns(self) -> int:
        """Warmup required only for entrypoints whose profile was cached."""

        return self._entrypoint_warmup_wall_time_ns

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
            # every unique ABI's CUDA examples alive until take_functions()
            # makes isolated profiling scale with the sum of model-stage
            # inputs, rather than the largest ABI. Retain only the executable.
            self._executables.release_occurrence_values(executable)
            return measurement
        finally:
            del executable

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

        torch.cuda.set_device(self._device_ordinal)
        stream = torch.cuda.current_stream(self._device_ordinal)
        phase_timings: list[tuple[str, int]] = []
        self._condition_profile_device(stream, phase_timings)
        persistent_baseline = self._requested_allocated_bytes()
        persistent_high_water = self._warm_profile_provider(
            executable,
            stream,
            baseline=persistent_baseline,
            phase_timings=phase_timings,
        )
        timing = self._collect_timing_samples(executable, stream, phase_timings)
        self._library.shadowspill_pytorch_allocator_wait_idle()
        persistent_high_water = max(
            persistent_high_water, self._requested_allocated_bytes()
        )
        workspace, persistent_high_water, allocation_abi = self._profile_workspace(
            executable,
            stream,
            persistent_high_water=persistent_high_water,
            phase_timings=phase_timings,
        )
        fixed_extents = _persistent_profile_extents(
            workspace,
            baseline=persistent_baseline,
            high_water=persistent_high_water,
        )
        return _task_measurement(
            executable,
            execution_provider,
            workspace,
            timing,
            fixed_extents,
            phase_timings,
            allocation_abi,
        )

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
            stream.synchronize()
            self._library.shadowspill_pytorch_allocator_wait_idle()
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
        phase_timings: list[tuple[str, int]],
    ) -> tuple[TaskWorkspaceProfile, int, TaskAllocationABI]:
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
        abi_started = time.perf_counter_ns()
        allocation_abi = self._validate_allocation_abi(
            executable,
            stream,
            workspace,
        )
        phase_timings.append(
            ("allocation_abi_validation", time.perf_counter_ns() - abi_started)
        )
        return workspace, high_water, allocation_abi

    def _validate_allocation_abi(
        self,
        executable: Callable[[], object],
        stream: torch.cuda.Stream,
        baseline: TaskWorkspaceProfile,
    ) -> TaskAllocationABI:
        """Require three independent pointer-free allocation traces to agree."""

        contract = (
            executable.artifact.storage_contract
            if isinstance(executable, ProfileExecutable)
            else None
        )
        expected = TaskAllocationABI.capture(
            baseline.allocation_abi_trace, contract
        )
        for repetition in range(2):
            observed = self._measure_workspace(executable, stream)
            candidate = TaskAllocationABI.capture(
                observed.allocation_abi_trace, contract
            )
            if candidate.compatibility_digest != expected.compatibility_digest:
                raise AllocationTelemetryError(
                    "compiled task allocation ABI changed across independent "
                    f"profiling traces (repetition={repetition + 2}, "
                    f"expected={expected.compatibility_digest}, "
                    f"observed={candidate.compatibility_digest})"
                )
            if observed.output_input_bindings != baseline.output_input_bindings:
                raise AllocationTelemetryError(
                    "compiled task output/input storage bindings changed across "
                    f"independent profiling traces (repetition={repetition + 2})"
                )
        return expected

    def _invoke_profile_task(
        self,
        executable: Callable[[], object],
        stream: torch.cuda.Stream,
    ) -> None:
        task_id = self._open_profile_task(stream)
        try:
            output = executable()
            del output
            self._close_profile_task(task_id, stream)
        except BaseException:
            self._library.shadowspill_pytorch_abort_task_range()
            raise

    def _measure_task_once(
        self,
        executable: Callable[[], object],
        stream: torch.cuda.Stream,
    ) -> int:
        start, finish = self._timing_event_pair()
        task_id = self._open_profile_task(stream)
        try:
            start.record(stream)
            output = executable()
            del output
            finish.record(stream)
            self._close_profile_task(task_id, stream)
        except BaseException:
            self._library.shadowspill_pytorch_abort_task_range()
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

    def _open_profile_task(self, stream: torch.cuda.Stream) -> int:
        task_id = self._next_task_id
        self._next_task_id += 1
        status = int(
            self._library.shadowspill_pytorch_before_task(
                task_id, stream.cuda_stream, None, 0, None, 0
            )
        )
        if status != 0:
            raise CaptureError(f"profiling before_task failed with status {status}")
        return task_id

    def _close_profile_task(self, task_id: int, stream: torch.cuda.Stream) -> None:
        status = int(
            self._library.shadowspill_pytorch_after_task(
                task_id, stream.cuda_stream, None, 0, None, 0
            )
        )
        if status != 0:
            raise CaptureError(f"profiling after_task failed with status {status}")

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
        context: str,
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
            f"allocator failed to become idle during {context}: {detail}"
        )

    def _measure_opaque_optimizer(
        self, artifact: OpaqueOptimizerArtifact
    ) -> TaskMeasurement:
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
            stream.synchronize()
            self._diagnose_allocator_idle(context=f"compiled entrypoint {digest}")
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
        task_id = self._next_task_id
        self._next_task_id += 1
        execution_started = time.perf_counter_ns()
        start_allocation_telemetry(self._library, capacity=self._telemetry_capacity)
        task_open = False
        output: object | None = None
        try:
            status = int(
                self._library.shadowspill_pytorch_before_task(
                    task_id, stream.cuda_stream, None, 0, None, 0
                )
            )
            if status != 0:
                raise CaptureError(f"profiling before_task failed with status {status}")
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
            # remain attributable to this ABI. The allocator retires their
            # physical ranges against the active compute stream.
            output = None
            status = int(
                self._library.shadowspill_pytorch_after_task(
                    task_id, stream.cuda_stream, None, 0, None, 0
                )
            )
            task_open = False
            if status != 0:
                raise CaptureError(f"profiling after_task failed with status {status}")
            stream.synchronize()
            self._library.shadowspill_pytorch_allocator_wait_idle()
        except BaseException:
            if task_open:
                self._library.shadowspill_pytorch_abort_task_range()
            raise
        finally:
            stop_allocation_telemetry(self._library)
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
        "ShadowSpill failed to profile structural ABI "
        f"{artifact.compatibility_digest} "
        f"(kind={kind}, operators=[{operator_text}]): {cause}",
        structural_abi=artifact.compatibility_digest,
        task_kind=kind,
        operators=operators,
    )


def _persistent_profile_extents(
    workspace: TaskWorkspaceProfile,
    *,
    baseline: int,
    high_water: int,
) -> tuple[int, ...]:
    # A shared provider cache may already be populated by another ABI. Keep at
    # least this task's rotating live set so a cached profile is conservative.
    fixed_bytes = max(
        0,
        high_water - baseline,
        sum(workspace.persistent_extent_bytes),
    )
    return () if fixed_bytes == 0 else (fixed_bytes,)


def _task_measurement(
    executable: Callable[[], object],
    execution_provider: str,
    workspace: TaskWorkspaceProfile,
    timing: _TimingObservation,
    fixed_extents: tuple[int, ...],
    phase_timings: list[tuple[str, int]],
    allocation_abi: TaskAllocationABI,
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
        allocation_abi=allocation_abi,
    )
