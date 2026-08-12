"""Exact accumulated-training dispatch through selected AOT graph pairs."""

from __future__ import annotations

import copy
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, cast

import torch
from torch.utils._pytree import tree_flatten, tree_map

from shadowspill.ir import ExecutionPlan, MemoryAction, MemoryActionKind, TaskSpec

from ._abi import AdapterStatistics
from ._runtime_trace import RuntimeTraceEvent, RuntimeTraceEventKind
from ._telemetry import CapturedAllocationEvent
from ._transfer_labels import TransferLabelIndex
from .capture import GraphArtifact
from .optimizer import OpaqueOptimizerArtifact, current_optimizer_bindings
from .runtime_bridge import RuntimeBridge, actions_by_task
from .training_lowering import LoweredTrainingProgram, TrainingTaskEntrypoint
from .training_materialization import TrainingMaterializedState


@dataclass(frozen=True, slots=True)
class _ExecutionTaskRecord:
    """Immutable, predecoded repeated-path view of one selected task."""

    entrypoint: TrainingTaskEntrypoint
    task: TaskSpec
    input_aliases: tuple[str, ...]
    actions: tuple[MemoryAction, ...]
    dense_task_id: int
    execution_ordinal: int
    semantic_name: str
    trace_label: str
    function: Callable[..., object] | None
    argument_template: tuple[object, ...] | None
    forward_outputs: tuple[_ForwardOutputRecord, ...]
    gradient_outputs: tuple[_GradientOutputRecord, ...]
    optimizer_argument_object_ids: tuple[str | None, ...]
    dematerialize_aliases: tuple[str, ...]
    released_ephemeral: tuple[tuple[str, tuple[str, ...]], ...]
    native_handle: int = 0


@dataclass(frozen=True, slots=True)
class _ForwardOutputRecord:
    """One predecoded forward leaf and its planned alias bundle."""

    leaf_index: int
    object_id: str
    alias_id: str
    adopt: bool
    replace: bool


@dataclass(frozen=True, slots=True)
class _GradientOutputRecord:
    """All contribution leaves accumulated into one planned gradient."""

    object_id: str
    alias_id: str
    leaf_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _PlanRun:
    lowered: LoweredTrainingProgram
    plan: ExecutionPlan
    expected_task_seconds: Mapping[str, float]
    execution: tuple[_ExecutionTaskRecord, ...]
    initial_device_aliases: tuple[str, ...]
    public_by_microbatch: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class _TensorLayout:
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    dtype: torch.dtype


@dataclass(frozen=True, slots=True)
class TaskExecutionTiming:
    """Qualification timing for one selected task invocation."""

    task_id: str
    execution_ordinal: int
    execution_task_id: str
    semantic_name: str
    phase: str
    microbatch: int | None
    expected_profile_seconds: float
    before_task_enter_timestamp_ns: int
    before_task_exit_timestamp_ns: int
    after_task_enter_timestamp_ns: int
    after_task_exit_timestamp_ns: int
    before_readiness_waits_timestamp_ns: int
    before_task_compute_timestamp_ns: int
    after_task_compute_timestamp_ns: int
    gpu_start_seconds: float
    gpu_end_seconds: float
    gpu_duration_seconds: float
    before_readiness_waits_seconds: float | None
    before_task_compute_seconds: float | None
    after_task_compute_seconds: float | None
    readiness_wait_seconds: float | None
    task_compute_seconds: float | None
    before_readiness_waits_sequence: int
    before_task_compute_sequence: int
    after_task_compute_sequence: int
    native_before_task_enter_seconds: float | None
    native_before_task_exit_seconds: float | None
    native_after_task_enter_seconds: float | None
    native_after_task_exit_seconds: float | None
    host_before_task_seconds: float
    host_stream_resolution_seconds: float
    host_readiness_marker_seconds: float
    host_native_before_task_seconds: float
    host_input_lookup_seconds: float
    host_storage_rebind_seconds: float
    host_generation_publish_seconds: float
    host_argument_assembly_seconds: float
    host_rebind_seconds: float
    host_dispatch_seconds: float
    host_output_flatten_seconds: float
    host_output_classification_seconds: float
    host_output_adoption_seconds: float
    host_output_state_publish_seconds: float
    host_gradient_accumulation_seconds: float
    host_output_publish_seconds: float
    host_dematerialize_seconds: float
    host_postprocess_seconds: float
    host_native_after_task_seconds: float
    host_cleanup_seconds: float
    host_after_task_seconds: float
    host_total_seconds: float

    def as_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "execution_ordinal": self.execution_ordinal,
            "execution_task_id": self.execution_task_id,
            "semantic_name": self.semantic_name,
            "phase": self.phase,
            "microbatch": self.microbatch,
            "expected_profile_seconds": self.expected_profile_seconds,
            "boundary_timestamps": {
                "unit": "nanoseconds",
                "host": {
                    "clock": "CLOCK_MONOTONIC",
                    "before_task": {
                        "enter": self.before_task_enter_timestamp_ns,
                        "exit": self.before_task_exit_timestamp_ns,
                    },
                    "after_task": {
                        "enter": self.after_task_enter_timestamp_ns,
                        "exit": self.after_task_exit_timestamp_ns,
                    },
                },
                "compute_stream": {
                    "clock": "cuda_event_elapsed_from_step_origin",
                    "before_readiness_waits": (
                        self.before_readiness_waits_timestamp_ns
                    ),
                    "before_task_compute": self.before_task_compute_timestamp_ns,
                    "after_task_compute": self.after_task_compute_timestamp_ns,
                },
            },
            "gpu_start_seconds": self.gpu_start_seconds,
            "gpu_end_seconds": self.gpu_end_seconds,
            "gpu_duration_seconds": self.gpu_duration_seconds,
            "before_readiness_waits_seconds": self.before_readiness_waits_seconds,
            "before_task_compute_seconds": self.before_task_compute_seconds,
            "after_task_compute_seconds": self.after_task_compute_seconds,
            "readiness_wait_seconds": self.readiness_wait_seconds,
            "task_compute_seconds": self.task_compute_seconds,
            "before_readiness_waits_sequence": (self.before_readiness_waits_sequence),
            "before_task_compute_sequence": self.before_task_compute_sequence,
            "after_task_compute_sequence": self.after_task_compute_sequence,
            "native_before_task_enter_seconds": (self.native_before_task_enter_seconds),
            "native_before_task_exit_seconds": self.native_before_task_exit_seconds,
            "native_after_task_enter_seconds": self.native_after_task_enter_seconds,
            "native_after_task_exit_seconds": self.native_after_task_exit_seconds,
            "host_before_task_seconds": self.host_before_task_seconds,
            "host_stream_resolution_seconds": self.host_stream_resolution_seconds,
            "host_readiness_marker_seconds": self.host_readiness_marker_seconds,
            "host_native_before_task_seconds": (self.host_native_before_task_seconds),
            "host_input_lookup_seconds": self.host_input_lookup_seconds,
            "host_storage_rebind_seconds": self.host_storage_rebind_seconds,
            "host_generation_publish_seconds": (self.host_generation_publish_seconds),
            "host_argument_assembly_seconds": self.host_argument_assembly_seconds,
            "host_rebind_seconds": self.host_rebind_seconds,
            "host_dispatch_seconds": self.host_dispatch_seconds,
            "host_output_flatten_seconds": self.host_output_flatten_seconds,
            "host_output_classification_seconds": (
                self.host_output_classification_seconds
            ),
            "host_output_adoption_seconds": self.host_output_adoption_seconds,
            "host_output_state_publish_seconds": (
                self.host_output_state_publish_seconds
            ),
            "host_gradient_accumulation_seconds": (
                self.host_gradient_accumulation_seconds
            ),
            "host_output_publish_seconds": self.host_output_publish_seconds,
            "host_dematerialize_seconds": self.host_dematerialize_seconds,
            "host_postprocess_seconds": self.host_postprocess_seconds,
            "host_native_after_task_seconds": self.host_native_after_task_seconds,
            "host_cleanup_seconds": self.host_cleanup_seconds,
            "host_after_task_seconds": self.host_after_task_seconds,
            "host_total_seconds": self.host_total_seconds,
        }


@dataclass(frozen=True, slots=True)
class ExecutionTiming:
    """Qualification-only decomposition of one accumulated training call."""

    compute_seconds: float
    optimizer_seconds: float
    host_call_seconds: float
    host_startup_wait_seconds: float
    host_initial_actions_seconds: float
    trace_setup_seconds: float
    phase_gpu_seconds: tuple[tuple[str, float], ...]
    tasks: Mapping[str, TaskExecutionTiming]

    def as_dict(self, *, include_tasks: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "compute_seconds": self.compute_seconds,
            "optimizer_seconds": self.optimizer_seconds,
            "host_call_seconds": self.host_call_seconds,
            "host_startup_wait_seconds": self.host_startup_wait_seconds,
            "host_initial_actions_seconds": self.host_initial_actions_seconds,
            "trace_setup_seconds": self.trace_setup_seconds,
            "phase_gpu_seconds": dict(self.phase_gpu_seconds),
        }
        if include_tasks:
            result["tasks"] = {
                execution_task_id: item.as_dict()
                for execution_task_id, item in self.tasks.items()
            }
        return result


@dataclass(frozen=True, slots=True)
class AllocatorTrace:
    """Ordered allocator lifetimes and before/after slab state."""

    events: tuple[CapturedAllocationEvent, ...]
    live_allocations_before: int
    live_allocations_after: int
    allocated_bytes_before: int
    allocated_bytes_after: int
    peak_allocated_bytes: int
    free_bytes_after: int
    largest_free_range_bytes_after: int
    external_fragmentation_bytes_after: int
    blocked_allocators_after: int
    overflow: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "events": [
                {
                    "sequence": item.sequence,
                    "task_id": item.task_id,
                    "allocation_id": item.allocation_id,
                    "generation": item.generation,
                    "requested_bytes": item.requested_bytes,
                    "charged_bytes": item.charged_bytes,
                    "slab_offset": item.slab_offset,
                    "kind": item.kind.name.lower(),
                    "category": item.category.name.lower(),
                }
                for item in self.events
            ],
            "live_allocations_before": self.live_allocations_before,
            "live_allocations_after": self.live_allocations_after,
            "allocated_bytes_before": self.allocated_bytes_before,
            "allocated_bytes_after": self.allocated_bytes_after,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "free_bytes_after": self.free_bytes_after,
            "largest_free_range_bytes_after": self.largest_free_range_bytes_after,
            "external_fragmentation_bytes_after": (
                self.external_fragmentation_bytes_after
            ),
            "blocked_allocators_after": self.blocked_allocators_after,
            "overflow": self.overflow,
        }


@dataclass(frozen=True, slots=True)
class TransferTrace:
    """Annotated actions and completed-transfer counter deltas."""

    actions: tuple[MemoryAction, ...]
    fetch_transfers: int
    evict_transfers: int
    bytes_fetched: int
    bytes_evicted: int
    events: tuple[RuntimeTraceEvent, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "actions": [item.to_dict() for item in self.actions],
            "fetch_transfers": self.fetch_transfers,
            "evict_transfers": self.evict_transfers,
            "bytes_fetched": self.bytes_fetched,
            "bytes_evicted": self.bytes_evicted,
            "events": [item.as_dict() for item in self.events],
        }


@dataclass(frozen=True, slots=True)
class RuntimeTrace:
    """Runtime counter changes and terminal queue state for the traced call."""

    wait_events_inserted: int
    allocation_requests: int
    zero_byte_allocation_requests: int
    materialized_allocation_requests: int
    free_requests: int
    record_stream_callbacks: int
    event_queries: int
    queued_actions_after: int
    pending_retirements_after: int
    callback_failures_after: int
    step_id: int
    begin_timestamp_ns: int
    end_timestamp_ns: int
    event_capacity: int
    allocation_event_capacity: int
    event_overflow: bool
    allocation_event_overflow: bool
    events: tuple[RuntimeTraceEvent, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "wait_events_inserted": self.wait_events_inserted,
            "allocation_requests": self.allocation_requests,
            "zero_byte_allocation_requests": self.zero_byte_allocation_requests,
            "materialized_allocation_requests": self.materialized_allocation_requests,
            "free_requests": self.free_requests,
            "record_stream_callbacks": self.record_stream_callbacks,
            "event_queries": self.event_queries,
            "queued_actions_after": self.queued_actions_after,
            "pending_retirements_after": self.pending_retirements_after,
            "callback_failures_after": self.callback_failures_after,
            "step_id": self.step_id,
            "begin_timestamp_ns": self.begin_timestamp_ns,
            "end_timestamp_ns": self.end_timestamp_ns,
            "event_capacity": self.event_capacity,
            "allocation_event_capacity": self.allocation_event_capacity,
            "event_overflow": self.event_overflow,
            "allocation_event_overflow": self.allocation_event_overflow,
            "events": [item.as_dict() for item in self.events],
        }


@dataclass(frozen=True, slots=True)
class SimulatorTaskComparison:
    execution_task_id: str
    task_id: str
    expected_profile_seconds: float
    observed_gpu_seconds: float
    delta_seconds: float

    def as_dict(self) -> dict[str, object]:
        return {
            "execution_task_id": self.execution_task_id,
            "task_id": self.task_id,
            "expected_profile_seconds": self.expected_profile_seconds,
            "observed_gpu_seconds": self.observed_gpu_seconds,
            "delta_seconds": self.delta_seconds,
        }


@dataclass(frozen=True, slots=True)
class StepDiagnostics:
    """Resolved, immutable detailed evidence for one real training call.

    The unresolved :class:`DiagnosticsHandle` is asynchronous. Resolving it is
    explicitly synchronizing because callback and CUDA-event records must have
    completed before they can be copied safely.
    """

    timing: ExecutionTiming
    tasks: Mapping[str, TaskExecutionTiming]
    allocator: AllocatorTrace
    transfers: TransferTrace
    runtime: RuntimeTrace
    simulator_comparison: Mapping[str, SimulatorTaskComparison]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "shadowspill.step_diagnostics/v1",
            "timing": self.timing.as_dict(include_tasks=False),
            "tasks": {
                execution_task_id: item.as_dict()
                for execution_task_id, item in self.tasks.items()
            },
            "allocator": self.allocator.as_dict(),
            "transfers": self.transfers.as_dict(),
            "runtime": self.runtime.as_dict(),
            "simulator_comparison": {
                execution_task_id: item.as_dict()
                for execution_task_id, item in self.simulator_comparison.items()
            },
        }


@dataclass(slots=True)
class _ArmedTaskTiming:
    entrypoint: TrainingTaskEntrypoint
    expected_profile_seconds: float
    execution_ordinal: int
    semantic_name: str
    readiness_event: torch.cuda.Event
    start_event: torch.cuda.Event
    end_event: torch.cuda.Event
    host_started_ns: int = 0
    host_finished_ns: int = 0
    host_before_finished_ns: int = 0
    host_after_started_ns: int = 0
    host_stream_resolution_ns: int = 0
    host_readiness_marker_ns: int = 0
    host_native_before_task_ns: int = 0
    host_input_lookup_ns: int = 0
    host_storage_rebind_ns: int = 0
    host_generation_publish_ns: int = 0
    host_argument_assembly_ns: int = 0
    host_rebind_ns: int = 0
    host_dispatch_ns: int = 0
    host_output_flatten_ns: int = 0
    host_output_classification_ns: int = 0
    host_output_adoption_ns: int = 0
    host_output_state_publish_ns: int = 0
    host_gradient_accumulation_ns: int = 0
    host_output_publish_ns: int = 0
    host_dematerialize_ns: int = 0
    host_postprocess_ns: int = 0
    host_native_after_task_ns: int = 0
    host_cleanup_ns: int = 0


@dataclass(slots=True)
class _ArmedExecutionTiming:
    origin_event: torch.cuda.Event
    start_event: torch.cuda.Event
    end_event: torch.cuda.Event
    tasks: dict[str, _ArmedTaskTiming]
    task_order: tuple[str, ...]
    started: bool = False
    finished: bool = False
    host_call_started_ns: int = 0
    host_call_finished_ns: int = 0
    host_startup_wait_ns: int = 0
    host_initial_actions_ns: int = 0
    stream: torch.cuda.Stream | None = None
    statistics_before: AdapterStatistics | None = None
    actions: tuple[MemoryAction, ...] = ()
    trace_setup_ns: int = 0


@dataclass(slots=True)
class _ArmedSpanTiming:
    """Two-event production-like selected-task timing bracket."""

    start_event: torch.cuda.Event
    end_event: torch.cuda.Event
    started: bool = False
    finished: bool = False


@dataclass(slots=True)
class _PreparedTask:
    run: _PlanRun
    record: _ExecutionTaskRecord
    stream: torch.cuda.Stream | None
    arguments: Sequence[object]
    function: Callable[..., object] | None
    eager_optimizer: bool
    timing: _ArmedTaskTiming | None
    runtime_scope_open: bool = True


@dataclass(frozen=True, slots=True)
class _ProcessedTaskOutputs:
    outputs: tuple[torch.Tensor, ...]
    adopted: tuple[tuple[torch.Tensor, str], ...]
    replacement_aliases: frozenset[str]


class TrainingExecutor:
    """Execute selected forward/backward variants and one optimizer update."""

    def __init__(
        self,
        initial: tuple[LoweredTrainingProgram, ExecutionPlan] | None,
        recurrent: tuple[LoweredTrainingProgram, ExecutionPlan],
        bridge: RuntimeBridge,
        state: TrainingMaterializedState,
        functions: dict[str, Callable[..., object]],
        optimizer: torch.optim.Optimizer,
        *,
        optimizer_state_preinitialized: bool = False,
        optimizer_state_was_lazy: bool = False,
    ) -> None:
        self._bridge = bridge
        self._state = state
        self._functions = functions
        self.optimizer = optimizer
        self._initial = None if initial is None else self._prepare(*initial)
        self._recurrent = self._prepare(*recurrent)
        if self._initial is None:
            self._recurrent = self._admit_run(self._recurrent)
        self._configure_task_trace_labels()
        self._gradients = {
            state.bridge.alias_for_object(item.gradient_object_id): model_parameter
            for item in recurrent[0].gradients
            for model_parameter in (state.model.get_parameter(item.parameter_name),)
        }
        self._invocations = 0
        self._optimizer_state_initialized = not optimizer_state_was_lazy
        self._optimizer_state_available = (
            optimizer_state_preinitialized or not optimizer_state_was_lazy
        )
        self._optimizer_size_by_alias = {
            item.alias_group_id: item.size_bytes
            for item in self._recurrent.plan.program.alias_groups
        }
        self._armed_execution_timing: _ArmedExecutionTiming | None = None
        self._armed_span_timing: _ArmedSpanTiming | None = None
        self._profiler_annotations_enabled = False
        self._trace_allocation_capacity = 1_000_000
        self._trace_event_capacity = max(
            65_536,
            64
            * max(
                len(self._recurrent.execution),
                len(self._initial.execution) if self._initial is not None else 0,
            ),
        )
        self._trace_start_event: torch.cuda.Event | None = None
        self._trace_end_event: torch.cuda.Event | None = None
        self._trace_origin_event: torch.cuda.Event | None = None
        self._trace_task_events: dict[
            str, tuple[torch.cuda.Event, torch.cuda.Event, torch.cuda.Event]
        ] = {}
        self._span_start_event: torch.cuda.Event | None = None
        self._span_end_event: torch.cuda.Event | None = None

    def _admit_run(self, run: _PlanRun) -> _PlanRun:
        labels = TransferLabelIndex(
            run.plan.program,
            {record.task.task_id: record.trace_label for record in run.execution},
        )
        admitted: list[_ExecutionTaskRecord] = []
        for record in run.execution:
            admitted.append(
                replace(
                    record,
                    native_handle=self._bridge.admit_execution(
                        record.task,
                        record.input_aliases,
                        record.actions,
                        labels.labels_for(record.actions),
                    ),
                )
            )
        return replace(run, execution=tuple(admitted))

    def prepare_execution_tracing(self) -> None:
        """Lazily allocate reusable trace buffers and timing events."""

        runs = tuple(run for run in (self._initial, self._recurrent) if run is not None)
        self._bridge.enable_debug_task_timing(
            record.task.task_id for run in runs for record in run.execution
        )
        self._bridge.disable_debug_task_timing()
        self._bridge.prepare_runtime_trace(
            event_capacity=self._trace_event_capacity,
            allocation_event_capacity=self._trace_allocation_capacity,
        )
        event_factory: Any = torch.cuda.Event
        task_ids = tuple(
            dict.fromkeys(
                record.task.task_id for run in runs for record in run.execution
            )
        )
        self._trace_origin_event = event_factory(enable_timing=True)
        self._trace_start_event = event_factory(enable_timing=True)
        self._trace_end_event = event_factory(enable_timing=True)
        self._trace_task_events = {
            task_id: (
                event_factory(enable_timing=True),
                event_factory(enable_timing=True),
                event_factory(enable_timing=True),
            )
            for task_id in task_ids
        }
        # PyTorch creates CUDA event handles lazily on first record. Force that
        # one-time setup before the real trace begins, then reuse every event.
        stream = torch.cuda.current_stream()
        self._trace_origin_event.record(stream)
        self._trace_start_event.record(stream)
        self._trace_end_event.record(stream)
        for readiness_event, start_event, end_event in self._trace_task_events.values():
            readiness_event.record(stream)
            start_event.record(stream)
            end_event.record(stream)
        stream.synchronize()

    def set_profiler_annotations(self, enabled: bool) -> None:
        """Toggle provider annotations independently of runtime tracing."""

        self._bridge.set_profiler_annotations(enabled)
        self._profiler_annotations_enabled = enabled

    def finish_profiler_annotations(self) -> None:
        """Drain annotated asynchronous work before disabling its provider."""

        if not self._profiler_annotations_enabled:
            return
        self._bridge.wait_idle()
        self.set_profiler_annotations(False)

    def __call__(
        self, inputs: Sequence[Sequence[Any]]
    ) -> tuple[tuple[torch.Tensor, ...], tuple[Any, ...]]:
        timing = self._armed_execution_timing
        if timing is not None:
            timing.host_call_started_ns = time.perf_counter_ns()
            timing.origin_event.record(torch.cuda.current_stream())
        if self._invocations:
            # V1 plans have a fresh terminal state. Preserve asynchronous
            # StepResult construction, but do not accidentally overlap the
            # next invocation with terminal transfers from the prior plan.
            started_ns = time.perf_counter_ns() if timing is not None else 0
            self._bridge.wait_idle()
            if timing is not None:
                timing.host_startup_wait_ns = time.perf_counter_ns() - started_ns
        run = (
            self._initial
            if self._initial is not None and not self._optimizer_state_initialized
            else self._recurrent
        )
        if run is None:
            raise AssertionError("initial optimizer plan is unavailable")
        self._state.refresh_inputs(inputs)
        started_ns = time.perf_counter_ns() if timing is not None else 0
        with self._profile_range("shadowspill.training.initial_actions"):
            self._bridge.submit_initial_actions(
                tuple(
                    MemoryAction("task_000000", alias_id, MemoryActionKind.PREFETCH)
                    for alias_id in run.initial_device_aliases
                ),
                task_number=(1 << 60) + self._invocations,
            )
        if timing is not None:
            timing.host_initial_actions_ns = time.perf_counter_ns() - started_ns
        public_tensors: dict[int, tuple[torch.Tensor, ...]] = {}
        for record in run.execution:
            entrypoint = record.entrypoint
            outputs = self._execute_task(run, record)
            if entrypoint.phase == "forward" and entrypoint.microbatch is not None:
                public_tensors[entrypoint.microbatch] = outputs[
                    : entrypoint.public_output_count
                ]
        ordered = tuple(public_tensors[index] for index in range(len(public_tensors)))
        aliases = tuple(
            alias_id for values in run.public_by_microbatch for alias_id in values
        )
        tensors = tuple(tensor for values in ordered for tensor in values)
        bindings = self._bridge.acquire_for_caller(
            aliases,
            tensors,
            task_number=(1 << 59) + self._invocations,
        )
        self._bridge.transfer_outputs_to_caller(aliases, tensors, bindings)
        for alias_id in aliases:
            self._state.object_store.pop(alias_id, None)
            self._state.generations.pop(alias_id, None)
        losses: list[torch.Tensor] = []
        metrics: list[Any] = []
        for capture, values in zip(self._state.captures, ordered, strict=True):
            losses.append(values[0].detach())
            metrics.append(
                capture.objective_schema.rebuild_metrics(
                    tuple(value.detach() for value in values[1:])
                )
            )
        self._invocations += 1
        if timing is not None:
            timing.host_call_finished_ns = time.perf_counter_ns()
        return tuple(losses), tuple(metrics)

    def arm_compute_timing(self, *, trace_setup_ns: int = 0) -> None:
        """Bracket the next invocation's numerical compute stream only.

        This qualification-only measurement begins after the first task's
        readiness waits and ends after the final optimizer launch. It excludes
        invocation-start staging and terminal writeback without synchronizing
        ordinary execution.
        """

        if self._armed_execution_timing is not None:
            raise RuntimeError("a compute timing measurement is already armed")
        run = (
            self._initial
            if self._initial is not None and not self._optimizer_state_initialized
            else self._recurrent
        )
        if run is None:
            raise AssertionError("timing cannot select an execution plan")
        if (
            self._trace_origin_event is None
            or self._trace_start_event is None
            or self._trace_end_event is None
        ):
            self.prepare_execution_tracing()
        origin_event = self._trace_origin_event
        start_event = self._trace_start_event
        end_event = self._trace_end_event
        if origin_event is None or start_event is None or end_event is None:
            raise AssertionError("trace event preparation did not complete")
        tasks = {
            record.task.task_id: _ArmedTaskTiming(
                record.entrypoint,
                run.expected_task_seconds[record.task.task_id],
                record.execution_ordinal,
                record.semantic_name,
                *self._trace_task_events[record.task.task_id],
            )
            for record in run.execution
        }
        armed = _ArmedExecutionTiming(
            origin_event,
            start_event,
            end_event,
            tasks,
            tuple(record.task.task_id for record in run.execution),
            statistics_before=self._bridge.statistics(),
            actions=(
                tuple(
                    MemoryAction("task_000000", alias_id, MemoryActionKind.PREFETCH)
                    for alias_id in run.initial_device_aliases
                )
                + run.plan.schedule.actions
            ),
            trace_setup_ns=trace_setup_ns,
        )
        self._bridge.enable_debug_task_timing(armed.task_order)
        try:
            self._bridge.begin_runtime_trace(step_id=self._invocations + 1)
        except BaseException:
            self._bridge.disable_debug_task_timing()
            raise
        self._armed_execution_timing = armed

    def arm_selected_span_timing(self) -> None:
        """Arm a two-event selected-task span without detailed tracing.

        This qualification path does not enable native tracing, callbacks,
        NVTX, per-task events, allocator snapshots, or Python component
        timestamps. The reusable events are materialized before arming so the
        measured call follows the ordinary production path.
        """

        if self._armed_execution_timing is not None:
            raise RuntimeError("detailed execution timing is already armed")
        if self._armed_span_timing is not None:
            raise RuntimeError("a selected-span measurement is already armed")
        if self._span_start_event is None or self._span_end_event is None:
            event_factory: Any = torch.cuda.Event
            self._span_start_event = event_factory(enable_timing=True)
            self._span_end_event = event_factory(enable_timing=True)
            stream = torch.cuda.current_stream()
            self._span_start_event.record(stream)
            self._span_end_event.record(stream)
            stream.synchronize()
        self._armed_span_timing = _ArmedSpanTiming(
            self._span_start_event,
            self._span_end_event,
        )

    def collect_selected_span_seconds(self) -> float:
        """Synchronize and return an armed production-like task span."""

        timing = self._armed_span_timing
        if timing is None or not timing.started:
            raise RuntimeError("no selected-span measurement has started")
        if not timing.finished:
            raise RuntimeError("the selected-span measurement has not finished")
        timing.end_event.synchronize()
        result = float(timing.start_event.elapsed_time(timing.end_event)) / 1_000.0
        self._armed_span_timing = None
        return result

    def collect_compute_seconds(self) -> float:
        """Synchronize and return the armed compute-only interval."""

        return self.collect_execution_timing().compute_seconds

    def collect_execution_timing(self) -> ExecutionTiming:
        """Synchronize and return one armed per-task timing decomposition."""

        return self.collect_step_diagnostics().timing

    def collect_step_diagnostics(self) -> StepDiagnostics:
        """Synchronize and resolve the structured trace for one real call."""

        timing = self._armed_execution_timing
        if timing is None or not timing.started:
            raise RuntimeError("no execution timing measurement has started")
        if not timing.finished:
            raise RuntimeError("the execution timing measurement has not finished")
        timing.end_event.synchronize()
        if timing.stream is None:
            raise RuntimeError("execution timing has no compute stream")
        # The end event precedes after_task's completion callback. Synchronize
        # the measured stream before reading callback-owned records.
        timing.stream.synchronize()
        try:
            callback_timings = self._bridge.read_debug_task_timing()
        finally:
            self._bridge.disable_debug_task_timing()
        # Resolving diagnostics is explicitly synchronizing. Drain terminal
        # transfers before closing the trace so queue, dispatch, on-wire, and
        # completion evidence all describe the same real invocation.
        self._bridge.wait_idle()
        native_trace = self._bridge.end_and_read_runtime_trace()
        allocation_events = native_trace.allocation_events
        statistics_after = self._bridge.statistics()
        statistics_before = timing.statistics_before
        if statistics_before is None:
            raise RuntimeError("execution timing omitted its initial statistics")
        callback_origin_ns = native_trace.begin_timestamp_ns
        task_results: list[TaskExecutionTiming] = []
        phase_seconds: dict[str, float] = {}
        for task_id in timing.task_order:
            task = timing.tasks[task_id]
            gpu_start = (
                float(timing.start_event.elapsed_time(task.start_event)) / 1_000.0
            )
            gpu_end = float(timing.start_event.elapsed_time(task.end_event)) / 1_000.0
            gpu_duration = (
                float(task.start_event.elapsed_time(task.end_event)) / 1_000.0
            )
            phase_seconds[task.entrypoint.phase] = (
                phase_seconds.get(task.entrypoint.phase, 0.0) + gpu_duration
            )
            callback = callback_timings.get(task_id)

            def relative(timestamp: int) -> float | None:
                return (
                    (timestamp - callback_origin_ns) / 1e9
                    if callback_origin_ns and timestamp
                    else None
                )

            before_readiness_waits_seconds = (
                float(timing.origin_event.elapsed_time(task.readiness_event)) / 1_000.0
            )
            before_task_compute_seconds = (
                float(timing.origin_event.elapsed_time(task.start_event)) / 1_000.0
            )
            after_task_compute_seconds = (
                float(timing.origin_event.elapsed_time(task.end_event)) / 1_000.0
            )
            before_readiness_waits_ns = int(before_readiness_waits_seconds * 1e9)
            before_task_compute_ns = int(before_task_compute_seconds * 1e9)
            after_task_compute_ns = int(after_task_compute_seconds * 1e9)
            sequence_base = task.execution_ordinal * 3
            task_results.append(
                TaskExecutionTiming(
                    task_id=task_id,
                    execution_ordinal=task.execution_ordinal,
                    execution_task_id=f"execution_{task.execution_ordinal:06d}",
                    semantic_name=task.semantic_name,
                    phase=task.entrypoint.phase,
                    microbatch=task.entrypoint.microbatch,
                    expected_profile_seconds=task.expected_profile_seconds,
                    before_task_enter_timestamp_ns=(
                        callback.before_task_enter_ns if callback is not None else 0
                    ),
                    before_task_exit_timestamp_ns=(
                        callback.before_task_exit_ns if callback is not None else 0
                    ),
                    after_task_enter_timestamp_ns=(
                        callback.after_task_enter_ns if callback is not None else 0
                    ),
                    after_task_exit_timestamp_ns=(
                        callback.after_task_exit_ns if callback is not None else 0
                    ),
                    before_readiness_waits_timestamp_ns=before_readiness_waits_ns,
                    before_task_compute_timestamp_ns=before_task_compute_ns,
                    after_task_compute_timestamp_ns=after_task_compute_ns,
                    gpu_start_seconds=gpu_start,
                    gpu_end_seconds=gpu_end,
                    gpu_duration_seconds=gpu_duration,
                    before_readiness_waits_seconds=before_readiness_waits_seconds,
                    before_task_compute_seconds=before_task_compute_seconds,
                    after_task_compute_seconds=after_task_compute_seconds,
                    readiness_wait_seconds=(
                        float(task.readiness_event.elapsed_time(task.start_event))
                        / 1_000.0
                    ),
                    task_compute_seconds=(
                        float(task.start_event.elapsed_time(task.end_event)) / 1_000.0
                    ),
                    before_readiness_waits_sequence=sequence_base + 1,
                    before_task_compute_sequence=sequence_base + 2,
                    after_task_compute_sequence=sequence_base + 3,
                    native_before_task_enter_seconds=relative(
                        callback.before_task_enter_ns if callback is not None else 0
                    ),
                    native_before_task_exit_seconds=relative(
                        callback.before_task_exit_ns if callback is not None else 0
                    ),
                    native_after_task_enter_seconds=relative(
                        callback.after_task_enter_ns if callback is not None else 0
                    ),
                    native_after_task_exit_seconds=relative(
                        callback.after_task_exit_ns if callback is not None else 0
                    ),
                    host_before_task_seconds=(
                        task.host_before_finished_ns - task.host_started_ns
                    )
                    / 1e9,
                    host_stream_resolution_seconds=(
                        task.host_stream_resolution_ns / 1e9
                    ),
                    host_readiness_marker_seconds=(task.host_readiness_marker_ns / 1e9),
                    host_native_before_task_seconds=(
                        task.host_native_before_task_ns / 1e9
                    ),
                    host_input_lookup_seconds=task.host_input_lookup_ns / 1e9,
                    host_storage_rebind_seconds=task.host_storage_rebind_ns / 1e9,
                    host_generation_publish_seconds=(
                        task.host_generation_publish_ns / 1e9
                    ),
                    host_argument_assembly_seconds=(
                        task.host_argument_assembly_ns / 1e9
                    ),
                    host_rebind_seconds=task.host_rebind_ns / 1e9,
                    host_dispatch_seconds=task.host_dispatch_ns / 1e9,
                    host_output_flatten_seconds=task.host_output_flatten_ns / 1e9,
                    host_output_classification_seconds=(
                        task.host_output_classification_ns / 1e9
                    ),
                    host_output_adoption_seconds=task.host_output_adoption_ns / 1e9,
                    host_output_state_publish_seconds=(
                        task.host_output_state_publish_ns / 1e9
                    ),
                    host_gradient_accumulation_seconds=(
                        task.host_gradient_accumulation_ns / 1e9
                    ),
                    host_output_publish_seconds=task.host_output_publish_ns / 1e9,
                    host_dematerialize_seconds=task.host_dematerialize_ns / 1e9,
                    host_postprocess_seconds=task.host_postprocess_ns / 1e9,
                    host_native_after_task_seconds=(
                        task.host_native_after_task_ns / 1e9
                    ),
                    host_cleanup_seconds=task.host_cleanup_ns / 1e9,
                    host_after_task_seconds=(
                        task.host_finished_ns - task.host_after_started_ns
                    )
                    / 1e9,
                    host_total_seconds=(task.host_finished_ns - task.host_started_ns)
                    / 1e9,
                )
            )
        optimizer = [item for item in task_results if item.phase == "optimizer"]
        optimizer_seconds = (
            max(item.gpu_end_seconds for item in optimizer)
            - min(item.gpu_start_seconds for item in optimizer)
            if optimizer
            else 0.0
        )
        task_results_by_execution_id = MappingProxyType(
            {item.execution_task_id: item for item in task_results}
        )
        execution_timing = ExecutionTiming(
            compute_seconds=float(timing.start_event.elapsed_time(timing.end_event))
            / 1_000.0,
            optimizer_seconds=optimizer_seconds,
            host_call_seconds=(
                timing.host_call_finished_ns - timing.host_call_started_ns
            )
            / 1e9,
            host_startup_wait_seconds=timing.host_startup_wait_ns / 1e9,
            host_initial_actions_seconds=timing.host_initial_actions_ns / 1e9,
            trace_setup_seconds=timing.trace_setup_ns / 1e9,
            phase_gpu_seconds=tuple(sorted(phase_seconds.items())),
            tasks=task_results_by_execution_id,
        )
        allocator = AllocatorTrace(
            events=allocation_events,
            live_allocations_before=int(statistics_before.runtime.live_allocations),
            live_allocations_after=int(statistics_after.runtime.live_allocations),
            allocated_bytes_before=int(statistics_before.runtime.allocated_bytes),
            allocated_bytes_after=int(statistics_after.runtime.allocated_bytes),
            peak_allocated_bytes=int(statistics_after.runtime.peak_allocated_bytes),
            free_bytes_after=int(statistics_after.runtime.free_bytes),
            largest_free_range_bytes_after=int(
                statistics_after.runtime.largest_free_range_bytes
            ),
            external_fragmentation_bytes_after=int(
                statistics_after.runtime.external_fragmentation_bytes
            ),
            blocked_allocators_after=int(statistics_after.runtime.blocked_allocators),
            overflow=bool(statistics_after.runtime.allocation_event_overflow),
        )
        transfers = TransferTrace(
            actions=timing.actions,
            fetch_transfers=int(
                statistics_after.runtime.fetch_transfers
                - statistics_before.runtime.fetch_transfers
            ),
            evict_transfers=int(
                statistics_after.runtime.evict_transfers
                - statistics_before.runtime.evict_transfers
            ),
            bytes_fetched=int(
                statistics_after.runtime.bytes_fetched
                - statistics_before.runtime.bytes_fetched
            ),
            bytes_evicted=int(
                statistics_after.runtime.bytes_evicted
                - statistics_before.runtime.bytes_evicted
            ),
            events=tuple(
                item
                for item in native_trace.events
                if item.kind
                in {
                    RuntimeTraceEventKind.ACTION_QUEUED,
                    RuntimeTraceEventKind.DESTINATION_RESERVED,
                    RuntimeTraceEventKind.TRANSFER_DISPATCHED,
                    RuntimeTraceEventKind.TRANSFER_COMPLETED,
                }
            ),
        )
        allocation_requests = int(
            statistics_after.allocation_callbacks
            - statistics_before.allocation_callbacks
        )
        zero_byte_allocation_requests = int(
            statistics_after.zero_size_allocation_callbacks
            - statistics_before.zero_size_allocation_callbacks
        )
        runtime = RuntimeTrace(
            wait_events_inserted=int(
                statistics_after.runtime.wait_events_inserted
                - statistics_before.runtime.wait_events_inserted
            ),
            allocation_requests=allocation_requests,
            zero_byte_allocation_requests=zero_byte_allocation_requests,
            materialized_allocation_requests=(
                allocation_requests - zero_byte_allocation_requests
            ),
            free_requests=int(
                statistics_after.free_callbacks - statistics_before.free_callbacks
            ),
            record_stream_callbacks=int(
                statistics_after.record_stream_callbacks
                - statistics_before.record_stream_callbacks
            ),
            event_queries=int(
                statistics_after.cuda.event_queries
                - statistics_before.cuda.event_queries
            ),
            queued_actions_after=int(statistics_after.runtime.queued_actions),
            pending_retirements_after=int(statistics_after.runtime.pending_retirements),
            callback_failures_after=int(statistics_after.callback_failures),
            step_id=native_trace.step_id,
            begin_timestamp_ns=native_trace.begin_timestamp_ns,
            end_timestamp_ns=native_trace.end_timestamp_ns,
            event_capacity=native_trace.event_capacity,
            allocation_event_capacity=native_trace.allocation_event_capacity,
            event_overflow=native_trace.event_overflow,
            allocation_event_overflow=native_trace.allocation_event_overflow,
            events=native_trace.events,
        )
        comparisons = MappingProxyType(
            {
                item.execution_task_id: SimulatorTaskComparison(
                    execution_task_id=item.execution_task_id,
                    task_id=item.task_id,
                    expected_profile_seconds=item.expected_profile_seconds,
                    observed_gpu_seconds=item.gpu_duration_seconds,
                    delta_seconds=(
                        item.gpu_duration_seconds - item.expected_profile_seconds
                    ),
                )
                for item in task_results
            }
        )
        self._armed_execution_timing = None
        return StepDiagnostics(
            timing=execution_timing,
            tasks=task_results_by_execution_id,
            allocator=allocator,
            transfers=transfers,
            runtime=runtime,
            simulator_comparison=comparisons,
        )

    def cancel_execution_timing(self) -> None:
        """Synchronously tear down an armed debug trace after execution failure."""

        timing = self._armed_execution_timing
        if timing is None:
            return
        stream = timing.stream or torch.cuda.current_stream()
        stream.synchronize()
        try:
            self._bridge.disable_debug_task_timing()
        finally:
            with suppress(BaseException):
                self._bridge.end_and_read_runtime_trace()
            self._armed_execution_timing = None

    def _record_compute_start(self, stream: torch.cuda.Stream | None) -> None:
        if stream is None:
            return
        span = self._armed_span_timing
        if span is not None and not span.started:
            span.start_event.record(stream)
            span.started = True
        timing = self._armed_execution_timing
        if timing is None or timing.started:
            return
        timing.start_event.record(stream)
        timing.started = True
        timing.stream = stream

    def _record_compute_end(self, stream: torch.cuda.Stream | None) -> None:
        if stream is None:
            return
        span = self._armed_span_timing
        if span is not None and not span.finished:
            span.end_event.record(stream)
            span.finished = True
        timing = self._armed_execution_timing
        if timing is None or timing.finished:
            return
        timing.end_event.record(stream)
        timing.finished = True

    def _begin_task_timing(
        self, entrypoint: TrainingTaskEntrypoint
    ) -> _ArmedTaskTiming | None:
        timing = self._armed_execution_timing
        if timing is None:
            return None
        task = timing.tasks[entrypoint.task_id]
        task.host_started_ns = time.perf_counter_ns()
        return task

    @staticmethod
    def _finish_task_timing(task: _ArmedTaskTiming | None) -> None:
        if task is None:
            return
        task.host_finished_ns = time.perf_counter_ns()

    @staticmethod
    def _record_task_readiness(
        task: _ArmedTaskTiming | None, stream: torch.cuda.Stream | None
    ) -> None:
        if task is not None:
            if stream is None:
                raise AssertionError("task timing omitted its CUDA stream")
            task.readiness_event.record(stream)

    @staticmethod
    def _record_task_start(
        task: _ArmedTaskTiming | None, stream: torch.cuda.Stream | None
    ) -> None:
        if task is not None:
            if stream is None:
                raise AssertionError("task timing omitted its CUDA stream")
            task.start_event.record(stream)
            task.host_before_finished_ns = time.perf_counter_ns()

    @staticmethod
    def _record_task_end(
        task: _ArmedTaskTiming | None, stream: torch.cuda.Stream | None
    ) -> None:
        if task is not None:
            if stream is None:
                raise AssertionError("task timing omitted its CUDA stream")
            task.end_event.record(stream)
            task.host_after_started_ns = time.perf_counter_ns()

    @contextmanager
    def _profile_range(self, name: str) -> Iterator[None]:
        if not self._profiler_annotations_enabled:
            yield
            return
        range_id = self._bridge.profile_range_begin(name)
        try:
            yield
        finally:
            self._bridge.profile_range_end(range_id)

    @property
    def optimizer_state_initialized(self) -> bool:
        return self._optimizer_state_initialized

    def set_optimizer_state_initialized(self, value: bool) -> None:
        """Select the recurrent plan after a checkpoint restores lazy state."""

        if value and self._initial is None:
            self._optimizer_state_initialized = True
            self._optimizer_state_available = True
            return
        self._optimizer_state_initialized = value
        self._optimizer_state_available = value

    def optimizer_state_dict(self) -> dict[str, object]:
        """Synchronously snapshot optimizer state without stale CUDA pointers."""

        if not self._optimizer_state_initialized:
            raw = self.optimizer.state_dict()
            return {
                "state": {},
                "param_groups": copy.deepcopy(raw["param_groups"]),
            }

        exposed = self._expose_optimizer_state_cpu()
        try:
            raw = self.optimizer.state_dict()
            return cast(
                dict[str, object],
                tree_map(
                    lambda value: (
                        value.detach().cpu().clone()
                        if isinstance(value, torch.Tensor)
                        else copy.deepcopy(value)
                    ),
                    raw,
                ),
            )
        finally:
            self._restore_optimizer_host_only(exposed)

    def load_optimizer_state(self, value: Mapping[str, object]) -> bool:
        """Load ordinary optimizer state, then adopt spillable CUDA tensors."""

        self._bridge.wait_idle()
        self.optimizer.load_state_dict(copy.deepcopy(dict(value)))
        current = self._current_optimizer_bindings()
        planned = self._recurrent.lowered.optimizer_objects
        present = {item.name for item in planned if item.name in current}
        required_created = {item.name for item in planned if item.created_on_first_step}
        if present and present != {item.name for item in planned}:
            missing = sorted({item.name for item in planned} - present)
            raise RuntimeError(
                f"optimizer checkpoint has incomplete planned state: {missing}"
            )
        initialized = not required_created or required_created.issubset(present)
        if not initialized:
            aliases = tuple(
                self._bridge.alias_for_object(item.object_id) for item in planned
            )
            self._bridge.unregister(aliases)
            for item in planned:
                alias_id = self._bridge.alias_for_object(item.object_id)
                self._state.object_store.pop(alias_id, None)
                self._state.generations.pop(alias_id, None)
                self._state.object_tensors.pop(item.object_id, None)
            self._optimizer_state_initialized = False
            return False

        bound: list[tuple[str, torch.Tensor, int]] = []
        for item in planned:
            tensor = current[item.name].tensor
            if not tensor.is_cuda:
                if tensor.device.type != "cpu":
                    raise RuntimeError(
                        f"spillable optimizer state {item.name!r} restored on "
                        f"unsupported device {tensor.device.type}"
                    )
                layout = _TensorLayout(
                    tuple(tensor.shape),
                    tuple(tensor.stride()),
                    int(tensor.storage_offset()),
                    tensor.dtype,
                )
                owner = torch.empty(
                    tensor.untyped_storage().nbytes(),
                    dtype=torch.uint8,
                    device=self._state.device,
                )
                destination = self._view(owner, layout)
                destination.copy_(tensor)
                tensor.data = destination
            alias_id = self._bridge.alias_for_object(item.object_id)
            binding = self._bridge.promote_output(alias_id, tensor)
            self._bridge.rebind(tensor, alias_id, binding)
            self._state.object_store[alias_id] = tensor
            self._state.object_tensors[item.object_id] = tensor
            self._state.generations[alias_id] = binding.generation
            bound.append((alias_id, tensor, binding.generation))
        actions: list[MemoryAction] = []
        for alias_id, tensor, generation in bound:
            self._bridge.dematerialize(tensor, alias_id, generation)
            actions.append(
                MemoryAction("task_000000", alias_id, MemoryActionKind.OFFLOAD)
            )
        self._bridge.submit_initial_actions(
            tuple(actions), task_number=(1 << 58) + self._invocations
        )
        self._bridge.wait_idle()
        self._optimizer_state_initialized = True
        return True

    def restore_optimizer_cpu(self) -> None:
        """Leave live optimizer state backed by ordinary CPU storage."""

        self._expose_optimizer_state_cpu()

    def _execute_task(
        self,
        run: _PlanRun,
        record: _ExecutionTaskRecord,
    ) -> tuple[torch.Tensor, ...]:
        prepared = self._before_task(run, record)
        try:
            raw_outputs = self._run_compiled_task(prepared)
            return self._after_task(prepared, raw_outputs)
        except BaseException as error:
            self._abort_task(prepared, error)
            raise
        finally:
            self._finish_task_timing(prepared.timing)

    def _before_task(
        self,
        run: _PlanRun,
        record: _ExecutionTaskRecord,
    ) -> _PreparedTask:
        """Acquire, rebind, and assemble one complete frontend task boundary."""

        prepared = self._prepare_task(run, record)
        try:
            self._record_compute_start(prepared.stream)
            self._record_task_start(prepared.timing, prepared.stream)
            return prepared
        except BaseException as error:
            self._abort_task(prepared, error)
            self._finish_task_timing(prepared.timing)
            raise

    def _prepare_task(
        self,
        run: _PlanRun,
        record: _ExecutionTaskRecord,
    ) -> _PreparedTask:
        """Perform preparation while keeping its trace before compute start."""

        entrypoint = record.entrypoint
        task_timing = self._begin_task_timing(entrypoint)
        task = record.task
        trace_label = record.trace_label
        with self._profile_range(f"shadowspill.before_task.{trace_label}"):
            started_ns = time.perf_counter_ns() if task_timing is not None else 0
            needs_python_stream = (
                task_timing is not None
                or self._armed_span_timing is not None
                or not record.native_handle
            )
            stream = torch.cuda.current_stream() if needs_python_stream else None
            if task_timing is not None:
                task_timing.host_stream_resolution_ns = (
                    time.perf_counter_ns() - started_ns
                )
            input_aliases = record.input_aliases
            runtime_scope_open = False
            try:
                started_ns = time.perf_counter_ns() if task_timing is not None else 0
                self._record_task_readiness(task_timing, stream)
                if task_timing is not None:
                    task_timing.host_readiness_marker_ns = (
                        time.perf_counter_ns() - started_ns
                    )
                started_ns = time.perf_counter_ns() if task_timing is not None else 0
                with self._profile_range(f"shadowspill.storage_rebind.{trace_label}"):
                    component_started_ns = (
                        time.perf_counter_ns() if task_timing is not None else 0
                    )
                    input_tensors: list[torch.Tensor] = []
                    for alias_id in input_aliases:
                        tensor = self._state.object_store.get(alias_id)
                        if tensor is None:
                            raise RuntimeError(
                                f"task input {alias_id!r} has no tensor binding"
                            )
                        input_tensors.append(tensor)
                    if task_timing is not None:
                        task_timing.host_input_lookup_ns = (
                            time.perf_counter_ns() - component_started_ns
                        )
                    component_started_ns = (
                        time.perf_counter_ns() if task_timing is not None else 0
                    )
                    try:
                        if record.native_handle:
                            generations = self._bridge.before_execution_and_acquire(
                                record.native_handle,
                                record.dense_task_id,
                                self._state.device.index or 0,
                                input_tensors,
                            )
                        else:
                            bindings = self._bridge.before_task(
                                task.task_id,
                                stream
                                if stream is not None
                                else torch.cuda.current_stream(),
                                input_aliases,
                            )
                            rebound = list(
                                zip(
                                    input_tensors,
                                    input_aliases,
                                    bindings,
                                    strict=True,
                                )
                            )
                            self._bridge.rebind_many(rebound)
                            generations = tuple(
                                binding.generation for binding in bindings
                            )
                    except RuntimeError as error:
                        states = self._bridge.input_failure_states(input_aliases)
                        detail = (
                            "; ".join(states)
                            if states
                            else "all snapshots device-ready"
                        )
                        raise RuntimeError(
                            f"{error}; input_states=[{detail}]"
                        ) from error
                    runtime_scope_open = True
                    if task_timing is not None:
                        task_timing.host_native_before_task_ns = (
                            time.perf_counter_ns() - component_started_ns
                        )
                    component_started_ns = (
                        time.perf_counter_ns() if task_timing is not None else 0
                    )
                    for alias_id, generation in zip(
                        input_aliases, generations, strict=True
                    ):
                        self._state.generations[alias_id] = generation
                    if task_timing is not None:
                        task_timing.host_generation_publish_ns = (
                            time.perf_counter_ns() - component_started_ns
                        )
                    component_started_ns = (
                        time.perf_counter_ns() if task_timing is not None else 0
                    )
                    artifact = entrypoint.artifact
                    eager_optimizer = entrypoint.phase == "optimizer" and (
                        isinstance(artifact, OpaqueOptimizerArtifact)
                        or not self._optimizer_state_available
                    )
                    function: Callable[..., object] | None = None
                    if entrypoint.phase == "optimizer":
                        arguments: Sequence[object] = ()
                        if not eager_optimizer:
                            if not isinstance(artifact, GraphArtifact):
                                raise RuntimeError(
                                    "optimizer task has no executable artifact"
                                )
                            if all(
                                object_id is not None
                                for object_id in record.optimizer_argument_object_ids
                            ):
                                try:
                                    arguments = tuple(
                                        self._state.object_tensors[object_id]
                                        for object_id in (
                                            record.optimizer_argument_object_ids
                                        )
                                        if object_id is not None
                                    )
                                except KeyError as exc:
                                    raise RuntimeError(
                                        f"optimizer object {exc.args[0]!r} is unbound"
                                    ) from exc
                            else:
                                current = self._current_optimizer_bindings()
                                try:
                                    arguments = tuple(
                                        current[name].tensor
                                        for name in (entrypoint.optimizer_binding_names)
                                    )
                                except KeyError as exc:
                                    raise RuntimeError(
                                        f"optimizer tensor {exc.args[0]!r} is unbound"
                                    ) from exc
                            function = record.function
                    else:
                        eager_optimizer = False
                        if not isinstance(artifact, GraphArtifact):
                            raise RuntimeError("graph task has no captured artifact")
                        if record.argument_template is None:
                            raise AssertionError("graph argument template is absent")
                        graph_arguments = list(record.argument_template)
                        for slot in entrypoint.input_slots:
                            graph_arguments[slot.leaf_index] = (
                                self._state.object_tensors[slot.object_id]
                            )
                        arguments = graph_arguments
                        function = record.function
                    if task_timing is not None:
                        task_timing.host_argument_assembly_ns = (
                            time.perf_counter_ns() - component_started_ns
                        )
                if task_timing is not None:
                    task_timing.host_rebind_ns = time.perf_counter_ns() - started_ns
                return _PreparedTask(
                    run=run,
                    record=record,
                    stream=stream,
                    arguments=arguments,
                    function=function,
                    eager_optimizer=eager_optimizer,
                    timing=task_timing,
                )
            except BaseException as error:
                if runtime_scope_open:
                    self._bridge.abort_task_after_failure(
                        f"prepare task {task.task_id}", error
                    )
                self._finish_task_timing(task_timing)
                raise

    def _run_compiled_task(self, prepared: _PreparedTask) -> object:
        """Dispatch only the numerical task represented by ``prepared``."""

        started_ns = time.perf_counter_ns() if prepared.timing is not None else 0
        with (
            self._profile_range(
                f"shadowspill.compiled_call.{prepared.record.trace_label}"
            ),
            torch.no_grad(),
        ):
            if prepared.eager_optimizer:
                raw_outputs: object = self.optimizer.step()
            else:
                if prepared.function is None:
                    raise AssertionError("compiled task function is unavailable")
                raw_outputs = prepared.function(*prepared.arguments)
        if prepared.timing is not None:
            prepared.timing.host_dispatch_ns = time.perf_counter_ns() - started_ns
        self._record_task_end(prepared.timing, prepared.stream)
        if prepared.record.task.task_id == prepared.run.lowered.optimizer_task_id:
            self._record_compute_end(prepared.stream)
        return raw_outputs

    def _after_task(
        self,
        prepared: _PreparedTask,
        raw_outputs: object,
    ) -> tuple[torch.Tensor, ...]:
        """Publish outputs, actions, and cleanup for one frontend task."""

        record = prepared.record
        with self._profile_range(f"shadowspill.after_task.{record.trace_label}"):
            started_ns = time.perf_counter_ns() if prepared.timing is not None else 0
            with self._profile_range(
                f"shadowspill.output_processing.{record.trace_label}"
            ):
                processed = self._process_task_outputs(prepared, raw_outputs)
                del raw_outputs
                dematerialized = self._dematerialization_tensors(record)
            if prepared.timing is not None:
                prepared.timing.host_postprocess_ns = (
                    time.perf_counter_ns() - started_ns
                )

            started_ns = time.perf_counter_ns() if prepared.timing is not None else 0
            with self._profile_range(
                f"shadowspill.runtime.after_task.{record.trace_label}"
            ):
                if record.native_handle:
                    generations = self._bridge.after_execution_and_update(
                        record.native_handle,
                        record.dense_task_id,
                        self._state.device.index or 0,
                        processed.adopted,
                        dematerialized,
                        replacement_aliases=processed.replacement_aliases,
                    )
                else:
                    generations = self._bridge.adopt_many(
                        processed.adopted,
                        replacement_aliases=processed.replacement_aliases,
                    )
                    pending_dematerialization: list[tuple[torch.Tensor, str, int]] = []
                    for tensor, alias_id in zip(
                        dematerialized,
                        record.dematerialize_aliases,
                        strict=True,
                    ):
                        generation = self._state.generations.get(alias_id)
                        if generation is None:
                            raise RuntimeError(
                                f"action references unbound generation {alias_id!r}"
                            )
                        pending_dematerialization.append((tensor, alias_id, generation))
                    self._bridge.dematerialize_many(tuple(pending_dematerialization))
                    self._bridge.after_task(
                        record.task.task_id,
                        prepared.stream
                        if prepared.stream is not None
                        else torch.cuda.current_stream(),
                        record.task.mutations,
                        record.actions,
                    )
            prepared.runtime_scope_open = False
            if prepared.timing is not None:
                prepared.timing.host_native_after_task_ns = (
                    time.perf_counter_ns() - started_ns
                )
            started_ns = time.perf_counter_ns() if prepared.timing is not None else 0
            for (tensor, alias_id), generation in zip(
                processed.adopted, generations, strict=True
            ):
                if alias_id in processed.replacement_aliases:
                    self._state.replace_alias_generation(
                        alias_id, tensor, generation
                    )
                else:
                    self._state.generations[alias_id] = generation
            if prepared.timing is not None:
                prepared.timing.host_output_state_publish_ns = (
                    time.perf_counter_ns() - started_ns
                )

            started_ns = time.perf_counter_ns() if prepared.timing is not None else 0
            with self._profile_range(f"shadowspill.cleanup.{record.trace_label}"):
                self._cleanup_after_task(prepared)
            if prepared.timing is not None:
                prepared.timing.host_cleanup_ns = time.perf_counter_ns() - started_ns
            return processed.outputs

    def _process_task_outputs(
        self,
        prepared: _PreparedTask,
        raw_outputs: object,
    ) -> _ProcessedTaskOutputs:
        outputs: tuple[torch.Tensor, ...] = ()
        adopted: tuple[tuple[torch.Tensor, str], ...] = ()
        replacement_aliases: frozenset[str] = frozenset()
        entrypoint = prepared.record.entrypoint
        timing = prepared.timing
        if entrypoint.phase == "optimizer":
            started_ns = time.perf_counter_ns() if timing is not None else 0
            if prepared.eager_optimizer and not self._optimizer_state_available:
                self._bind_created_optimizer_state(prepared.run.lowered)
                self._optimizer_state_available = True
            if timing is not None:
                timing.host_output_publish_ns = time.perf_counter_ns() - started_ns
        else:
            started_ns = time.perf_counter_ns() if timing is not None else 0
            leaves, _ = tree_flatten(raw_outputs)
            if timing is not None:
                timing.host_output_flatten_ns = time.perf_counter_ns() - started_ns
            started_ns = time.perf_counter_ns() if timing is not None else 0
            if entrypoint.phase == "forward":
                tensor_outputs = tuple(
                    value for value in leaves if isinstance(value, torch.Tensor)
                )
                if len(tensor_outputs) != len(leaves):
                    raise RuntimeError("captured forward graph returned a static leaf")
                adopted, replacement_aliases = self._bind_forward_outputs(
                    prepared.record,
                    tensor_outputs,
                    timing,
                )
                outputs = tuple(
                    tensor_outputs[index]
                    for index in entrypoint.public_output_leaves
                )
            else:
                adopted = self._accumulate_gradients(prepared.record, leaves, timing)
            if timing is not None:
                timing.host_output_publish_ns = time.perf_counter_ns() - started_ns
            del leaves
        return _ProcessedTaskOutputs(outputs, adopted, replacement_aliases)

    def _cleanup_after_task(self, prepared: _PreparedTask) -> None:
        self._forget_released_objects(prepared.run, prepared.record)
        if prepared.record.task.task_id == prepared.run.lowered.optimizer_task_id:
            self._optimizer_state_initialized = True
            for parameter in self._gradients.values():
                parameter.grad = None
            for alias_id in self._gradients:
                self._state.object_store.pop(alias_id, None)
                self._state.generations.pop(alias_id, None)
            for gradient_binding in prepared.run.lowered.gradients:
                self._state.object_tensors.pop(
                    gradient_binding.gradient_object_id, None
                )

    def _abort_task(
        self,
        prepared: _PreparedTask,
        error: BaseException,
    ) -> None:
        if prepared.runtime_scope_open:
            prepared.runtime_scope_open = False
            self._bridge.abort_task_after_failure(
                f"execute task {prepared.record.task.task_id}", error
            )

    def _bind_forward_outputs(
        self,
        record: _ExecutionTaskRecord,
        outputs: tuple[torch.Tensor, ...],
        timing: _ArmedTaskTiming | None,
    ) -> tuple[tuple[tuple[torch.Tensor, str], ...], frozenset[str]]:
        started_ns = time.perf_counter_ns() if timing is not None else 0
        adopted: list[tuple[torch.Tensor, str]] = []
        replacements: set[str] = set()
        for item in record.forward_outputs:
            tensor = outputs[item.leaf_index]
            if item.adopt:
                adopted.append((tensor, item.alias_id))
            if item.replace:
                replacements.add(item.alias_id)
            else:
                self._state.object_store.setdefault(item.alias_id, tensor)
                self._state.object_tensors[item.object_id] = tensor
        if timing is not None:
            timing.host_output_classification_ns = time.perf_counter_ns() - started_ns
        return tuple(adopted), frozenset(replacements)

    def _accumulate_gradients(
        self,
        record: _ExecutionTaskRecord,
        leaves: list[object],
        timing: _ArmedTaskTiming | None,
    ) -> tuple[tuple[torch.Tensor, str], ...]:
        started_ns = time.perf_counter_ns() if timing is not None else 0
        contributions: list[torch.Tensor] = []
        destinations: list[torch.Tensor] = []
        first: list[tuple[str, str, torch.Tensor]] = []
        for item in record.gradient_outputs:
            values = [leaves[index] for index in item.leaf_indices]
            if not all(isinstance(value, torch.Tensor) for value in values):
                raise RuntimeError("parameter gradient became non-tensor")
            contribution = cast(torch.Tensor, values[0])
            for additional in values[1:]:
                if not isinstance(additional, torch.Tensor):
                    raise AssertionError("validated gradient became non-tensor")
                contribution.add_(additional)
            destination = self._state.object_store.get(item.alias_id)
            if destination is None:
                first.append((item.object_id, item.alias_id, contribution))
            elif _same_tensor_view(destination, contribution):
                self._state.object_tensors[item.object_id] = destination
                parameter = self._gradients.get(item.alias_id)
                if parameter is not None:
                    parameter.grad = destination
            else:
                destinations.append(destination)
                contributions.append(contribution)
        if timing is not None:
            timing.host_output_classification_ns = time.perf_counter_ns() - started_ns
        adopted: list[tuple[torch.Tensor, str]] = []
        for _object_id, alias_id, contribution in first:
            adopted.append((contribution, alias_id))
        started_ns = time.perf_counter_ns() if timing is not None else 0
        for object_id, alias_id, contribution in first:
            self._state.object_store[alias_id] = contribution
            self._state.object_tensors[object_id] = contribution
            parameter = self._gradients.get(alias_id)
            if parameter is not None:
                parameter.grad = contribution
        if timing is not None:
            timing.host_output_state_publish_ns = time.perf_counter_ns() - started_ns
        if destinations:
            started_ns = time.perf_counter_ns() if timing is not None else 0
            torch._foreach_add_(destinations, contributions)
            if timing is not None:
                timing.host_gradient_accumulation_ns = (
                    time.perf_counter_ns() - started_ns
                )
        return tuple(adopted)

    def _bind_created_optimizer_state(self, lowered: LoweredTrainingProgram) -> None:
        current = self._current_optimizer_bindings()
        produced: set[str] = set()
        adopted: list[tuple[torch.Tensor, str]] = []
        bound: list[tuple[str, torch.Tensor, str]] = []
        for item in lowered.optimizer_objects:
            if not item.created_on_first_step:
                continue
            actual = current.get(item.name)
            if actual is None:
                raise RuntimeError(
                    f"optimizer did not create planned state {item.name!r}"
                )
            tensor = actual.tensor
            if not tensor.is_cuda:
                if tensor.device.type != "cpu":
                    raise RuntimeError(
                        f"spillable optimizer state {item.name!r} was created on "
                        f"unsupported device {tensor.device.type}"
                    )
                layout = _TensorLayout(
                    tuple(tensor.shape),
                    tuple(tensor.stride()),
                    int(tensor.storage_offset()),
                    tensor.dtype,
                )
                owner = torch.empty(
                    tensor.untyped_storage().nbytes(),
                    dtype=torch.uint8,
                    device=self._state.device,
                )
                destination = self._view(owner, layout)
                destination.copy_(tensor)
                tensor.data = destination
            alias_id = self._bridge.alias_for_object(item.object_id)
            if alias_id not in produced:
                adopted.append((tensor, alias_id))
                bound.append((item.object_id, tensor, alias_id))
                produced.add(alias_id)
            else:
                self._state.object_tensors[item.object_id] = tensor
        generations = self._bridge.adopt_many(adopted)
        for (object_id, tensor, alias_id), generation in zip(
            bound, generations, strict=True
        ):
            self._state.object_store[alias_id] = tensor
            self._state.object_tensors[object_id] = tensor
            self._state.generations[alias_id] = generation

    def _current_optimizer_bindings(self) -> dict[str, Any]:
        return {
            item.name: item
            for item in current_optimizer_bindings(
                dict(self._state.model.named_parameters()), self.optimizer
            )
        }

    def _expose_optimizer_state_cpu(
        self,
    ) -> tuple[tuple[str, torch.Tensor, _TensorLayout], ...]:
        self._bridge.wait_idle()
        current = self._current_optimizer_bindings()
        exposed: list[tuple[str, torch.Tensor, _TensorLayout]] = []
        owners: dict[str, torch.Tensor] = {}
        for item in self._recurrent.lowered.optimizer_objects:
            actual = current.get(item.name)
            if actual is None:
                continue
            tensor = actual.tensor
            alias_id = self._bridge.alias_for_object(item.object_id)
            owner = owners.get(alias_id)
            if owner is None:
                owner = torch.empty(
                    self._optimizer_size_by_alias[alias_id],
                    dtype=torch.uint8,
                    device="cpu",
                )
                self._bridge.read_spill_tensor(alias_id, owner)
                owners[alias_id] = owner
            layout = _TensorLayout(
                tuple(tensor.shape),
                tuple(tensor.stride()),
                int(tensor.storage_offset()),
                tensor.dtype,
            )
            tensor.data = self._view(owner, layout)
            exposed.append((alias_id, tensor, layout))
        return tuple(exposed)

    def _restore_optimizer_host_only(
        self, exposed: tuple[tuple[str, torch.Tensor, _TensorLayout], ...]
    ) -> None:
        if not exposed:
            return
        owners: dict[str, torch.Tensor] = {}
        released: set[str] = set()
        actions: list[MemoryAction] = []
        for alias_id, tensor, layout in exposed:
            owner = owners.get(alias_id)
            if owner is None:
                owner = torch.empty(
                    self._optimizer_size_by_alias[alias_id],
                    dtype=torch.uint8,
                    device=self._state.device,
                )
                owners[alias_id] = owner
            tensor.data = self._view(owner, layout)
            if alias_id in released:
                continue
            binding = self._bridge.bind_registered_tensor(alias_id, owner)
            self._bridge.rebind(tensor, alias_id, binding)
            self._state.object_store[alias_id] = tensor
            self._state.generations[alias_id] = binding.generation
            self._bridge.dematerialize(tensor, alias_id, binding.generation)
            actions.append(
                MemoryAction("task_000000", alias_id, MemoryActionKind.RELEASE)
            )
            released.add(alias_id)
        self._bridge.submit_initial_actions(
            tuple(actions), task_number=(1 << 57) + self._invocations
        )
        self._bridge.wait_idle()

    @staticmethod
    def _view(owner: torch.Tensor, layout: _TensorLayout) -> torch.Tensor:
        return torch.empty(0, dtype=layout.dtype, device=owner.device).set_(
            owner.untyped_storage(),
            layout.storage_offset,
            layout.shape,
            layout.stride,
        )

    def _dematerialization_tensors(
        self, record: _ExecutionTaskRecord
    ) -> tuple[torch.Tensor, ...]:
        pending: list[torch.Tensor] = []
        for alias_id in record.dematerialize_aliases:
            tensor = self._state.object_store.get(alias_id)
            if tensor is None:
                raise RuntimeError(f"action references unbound object {alias_id!r}")
            pending.append(tensor)
        return tuple(pending)

    def _forget_released_objects(
        self, run: _PlanRun, record: _ExecutionTaskRecord
    ) -> None:
        del run
        for alias_id, object_ids in record.released_ephemeral:
            self._state.object_store.pop(alias_id, None)
            self._state.generations.pop(alias_id, None)
            for object_id in object_ids:
                self._state.object_tensors.pop(object_id, None)

    def _prepare(
        self, lowered: LoweredTrainingProgram, plan: ExecutionPlan
    ) -> _PlanRun:
        tasks = {
            item.task_id: item for item in plan.program.selected_tasks(plan.selections)
        }
        profiles = {item.profile_id: item for item in plan.program.profiles}
        entrypoints = tuple(
            item for item in lowered.entrypoints if item.task_id in tasks
        )
        actions = actions_by_task(plan.schedule.actions)
        input_aliases_by_task = {
            task_id: tuple(
                dict.fromkeys(
                    self._bridge.alias_for_object(object_id)
                    for object_id in task.inputs
                )
            )
            for task_id, task in tasks.items()
        }
        initial_aliases = {
            item.alias_group_id for item in plan.schedule.initial_residency
        }
        ephemeral_aliases = frozenset(
            item.alias_group_id
            for item in plan.program.alias_groups
            if item.alias_group_id not in initial_aliases
        )
        objects_by_alias = {
            alias_id: tuple(
                item.object_id
                for item in plan.program.objects
                if item.alias_group_id == alias_id
            )
            for alias_id in (item.alias_group_id for item in plan.program.alias_groups)
        }
        optimizer_object_by_name = {
            **{
                item.parameter_name: item.parameter_object_id
                for item in lowered.gradients
            },
            **{
                f"gradient.{item.parameter_name}": item.gradient_object_id
                for item in lowered.gradients
            },
            **{item.name: item.object_id for item in lowered.optimizer_objects},
        }
        identities = _selected_entrypoint_identities(entrypoints)
        execution: list[_ExecutionTaskRecord] = []
        for entrypoint in entrypoints:
            artifact = entrypoint.artifact
            function = (
                self._functions[artifact.compatibility_digest]
                if isinstance(artifact, GraphArtifact)
                else None
            )
            argument_template = (
                tuple(artifact.example_arguments)
                if isinstance(artifact, GraphArtifact)
                and entrypoint.phase != "optimizer"
                else None
            )
            execution_ordinal, semantic_name = identities[entrypoint.task_id]
            input_aliases = input_aliases_by_task[entrypoint.task_id]
            produced: set[str] = set()
            forward_outputs: list[_ForwardOutputRecord] = []
            replacement_leaves = set(entrypoint.replacement_output_leaves)
            for slot in entrypoint.output_slots:
                alias_id = self._bridge.alias_for_object(slot.object_id)
                replace_output = slot.leaf_index in replacement_leaves
                adopt = (
                    replace_output or alias_id not in input_aliases
                ) and alias_id not in produced
                if adopt:
                    produced.add(alias_id)
                forward_outputs.append(
                    _ForwardOutputRecord(
                        slot.leaf_index,
                        slot.object_id,
                        alias_id,
                        adopt,
                        replace_output,
                    )
                )
            gradient_by_alias: dict[str, tuple[str, list[int]]] = {}
            for slot in entrypoint.gradient_output_slots:
                alias_id = self._bridge.alias_for_object(slot.object_id)
                group = gradient_by_alias.setdefault(alias_id, (slot.object_id, []))
                group[1].append(slot.leaf_index)
            gradient_outputs = tuple(
                _GradientOutputRecord(object_id, alias_id, tuple(indices))
                for alias_id, (object_id, indices) in gradient_by_alias.items()
            )
            optimizer_argument_object_ids = tuple(
                optimizer_object_by_name.get(name)
                for name in entrypoint.optimizer_binding_names
            )
            task_actions = actions.get(entrypoint.task_id, ())
            dematerialize_aliases = tuple(
                item.alias_group_id
                for item in task_actions
                if item.kind
                in {
                    MemoryActionKind.RELEASE,
                    MemoryActionKind.OFFLOAD,
                }
            )
            released_ephemeral = tuple(
                (item.alias_group_id, objects_by_alias[item.alias_group_id])
                for item in task_actions
                if item.kind is MemoryActionKind.RELEASE
                and item.alias_group_id in ephemeral_aliases
            )
            execution.append(
                _ExecutionTaskRecord(
                    entrypoint=entrypoint,
                    task=tasks[entrypoint.task_id],
                    input_aliases=input_aliases,
                    actions=task_actions,
                    dense_task_id=int(entrypoint.task_id.removeprefix("task_")),
                    execution_ordinal=execution_ordinal,
                    semantic_name=semantic_name,
                    trace_label=(f"execution_{execution_ordinal:06d}.{semantic_name}"),
                    function=function,
                    argument_template=argument_template,
                    forward_outputs=tuple(forward_outputs),
                    gradient_outputs=gradient_outputs,
                    optimizer_argument_object_ids=optimizer_argument_object_ids,
                    dematerialize_aliases=dematerialize_aliases,
                    released_ephemeral=released_ephemeral,
                )
            )
        return _PlanRun(
            lowered=lowered,
            plan=plan,
            expected_task_seconds={
                task_id: profiles[task.profile_id].runtime_ns / 1e9
                for task_id, task in tasks.items()
            },
            execution=tuple(execution),
            initial_device_aliases=tuple(
                item.alias_group_id
                for item in plan.schedule.initial_residency
                if item.location.value == "device"
            ),
            public_by_microbatch=self._public_outputs(entrypoints),
        )

    def _configure_task_trace_labels(self) -> dict[str, str]:
        """Register chronological semantic labels once, before execution."""

        result: dict[str, str] = {}
        for run in (self._initial, self._recurrent):
            if run is None:
                continue
            for record in run.execution:
                previous = result.setdefault(record.task.task_id, record.trace_label)
                if previous != record.trace_label:
                    raise RuntimeError(
                        f"task {record.task.task_id} has conflicting trace labels "
                        f"{previous!r} and {record.trace_label!r}"
                    )
        self._bridge.configure_task_labels(result)
        return result

    def _public_outputs(
        self, entrypoints: tuple[TrainingTaskEntrypoint, ...]
    ) -> tuple[tuple[str, ...], ...]:
        result: dict[int, tuple[str, ...]] = {}
        for entrypoint in entrypoints:
            if entrypoint.phase != "forward" or entrypoint.microbatch is None:
                continue
            result[entrypoint.microbatch] = tuple(
                self._bridge.alias_for_object(
                    next(
                        slot.object_id
                        for slot in entrypoint.output_slots
                        if slot.leaf_index == leaf_index
                    )
                )
                for leaf_index in entrypoint.public_output_leaves
            )
        return tuple(result[index] for index in range(len(result)))


def _same_tensor_view(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Return whether two tensors name the same bytes with the same geometry."""

    return bool(
        left.untyped_storage()._cdata == right.untyped_storage()._cdata
        and left.storage_offset() == right.storage_offset()
        and left.shape == right.shape
        and left.stride() == right.stride()
        and left.dtype == right.dtype
    )


def _selected_entrypoint_identities(
    entrypoints: tuple[TrainingTaskEntrypoint, ...],
) -> dict[str, tuple[int, str]]:
    """Assign dense execution ordinals and readable semantic task names."""

    result: dict[str, tuple[int, str]] = {}
    phase_ordinals: dict[str, int] = {}
    for execution_ordinal, entrypoint in enumerate(entrypoints):
        if entrypoint.microbatch is not None and entrypoint.stage_index is not None:
            semantic_name = (
                f"microbatch_{entrypoint.microbatch:04d}."
                f"stage_{entrypoint.stage_index:04d}."
                f"{entrypoint.phase}.{entrypoint.variant}"
            )
        else:
            phase_ordinal = phase_ordinals.get(entrypoint.phase, 0)
            phase_ordinals[entrypoint.phase] = phase_ordinal + 1
            semantic_name = f"{entrypoint.phase}.component_{phase_ordinal:04d}"
        result[entrypoint.task_id] = (execution_ordinal, semantic_name)
    return result


__all__ = ["ExecutionTiming", "TaskExecutionTiming", "TrainingExecutor"]
