"""Exact accumulated-training dispatch through selected AOT graph pairs."""

from __future__ import annotations

import copy
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, suppress
from dataclasses import dataclass, replace
from typing import Any, cast

import torch
from torch.utils._pytree import tree_flatten, tree_map

from shadowspill.ir import ExecutionPlan, MemoryAction, MemoryActionKind
from shadowspill.planner.diagnostics.mapping import FrozenMapping
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.diagnostics.collection import collect_step_diagnostics
from shadowspill.pytorch.diagnostics.execution import (
    StepDiagnostics,
)
from shadowspill.pytorch.diagnostics.timing import (
    ArmedExecutionTiming as _ArmedExecutionTiming,
)
from shadowspill.pytorch.diagnostics.timing import (
    ArmedSpanTiming as _ArmedSpanTiming,
)
from shadowspill.pytorch.diagnostics.timing import (
    ArmedTaskTiming as _ArmedTaskTiming,
)
from shadowspill.pytorch.invocation import ReusableCompletionEvent
from shadowspill.pytorch.lowering.training import (
    LoweredTrainingProgram,
    TrainingTaskEntrypoint,
)
from shadowspill.pytorch.materialization.replacement import ReplacementStorageViews
from shadowspill.pytorch.materialization.training import TrainingMaterializedState
from shadowspill.pytorch.optimizer import (
    OpaqueOptimizerArtifact,
    current_optimizer_bindings,
    opaque_optimizer_outputs,
    restore_optimizer_checkpoint_structure,
)
from shadowspill.pytorch.runtime_adapter.bridge import (
    PublishedStorage,
    RuntimeBridge,
    TaskMemoryEnvelope,
)
from shadowspill.pytorch.runtime_adapter.failures import (
    allocator_oom_error,
    generic_runtime_error,
    read_allocator_failure,
)
from shadowspill.pytorch.runtime_adapter.fixed_layout import RuntimeFixedLayout
from shadowspill.pytorch.runtime_adapter.transfer_labels import TransferLabelIndex
from shadowspill.pytorch.state.optimizer import release_optimizer_state_from_plan
from shadowspill.simulator import SimulationResult

from .annotations import AnnotatedExecutor, TaskBoundaryAnnotations
from .records import (
    ExecutionTaskRecord as _ExecutionTaskRecord,
)
from .records import (
    PlanRun as _PlanRun,
)
from .records import (
    build_plan_run,
)


@dataclass(frozen=True, slots=True)
class _TensorLayout:
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    dtype: torch.dtype


@dataclass(frozen=True, slots=True)
class _ExposedOptimizerTensor:
    tensor: torch.Tensor
    device_placeholder: torch.Tensor


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
class _TaskCall:
    arguments: Sequence[object]
    function: Callable[..., object] | None
    eager_optimizer: bool


@dataclass(frozen=True, slots=True)
class _ProcessedTaskOutputs:
    outputs: tuple[torch.Tensor, ...]
    adopted: tuple[PublishedStorage, ...]
    replacements: tuple[ReplacementStorageViews, ...]
    optimizer_bindings: tuple[tuple[str, torch.Tensor, str], ...] = ()

    @property
    def replacement_aliases(self) -> frozenset[str]:
        return frozenset(item.alias_id for item in self.replacements)


def _alias_accesses(
    run: _PlanRun,
) -> FrozenMapping[str, tuple[tuple[int, bool], ...]]:
    """Each alias group's reads and writes by the selected tasks, in order."""

    alias_of = {
        item.object_id: item.alias_group_id for item in run.plan.program.objects
    }
    accesses: dict[str, list[tuple[int, bool]]] = {}
    for record in run.execution:
        task = record.task
        for object_id in task.inputs:
            accesses.setdefault(alias_of[object_id], []).append(
                (record.execution_ordinal, False)
            )
        written = tuple(task.outputs) + tuple(item.object_id for item in task.mutations)
        for object_id in written:
            accesses.setdefault(alias_of[object_id], []).append(
                (record.execution_ordinal, True)
            )
    return FrozenMapping({key: tuple(value) for key, value in accesses.items()})


class TrainingExecutor(AnnotatedExecutor):
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
        recurrent_simulation: SimulationResult,
        initial_simulation: SimulationResult | None = None,
        initial_fixed_layout: RuntimeFixedLayout | None = None,
        recurrent_fixed_layout: RuntimeFixedLayout,
        initial_memory_envelopes: Mapping[str, TaskMemoryEnvelope] | None = None,
        recurrent_memory_envelopes: Mapping[str, TaskMemoryEnvelope],
        optimizer_state_preinitialized: bool = False,
        optimizer_state_was_lazy: bool = False,
    ) -> None:
        self._bridge = bridge
        self._state = state
        self._functions = functions
        self.optimizer = optimizer
        if initial is not None and initial_simulation is None:
            raise ValueError(
                "initial execution plan requires matching simulator evidence"
            )
        self._initial = (
            None
            if initial is None
            else build_plan_run(
                *initial,
                simulation=cast(SimulationResult, initial_simulation),
                bridge=bridge,
                functions=functions,
                memory_envelopes=initial_memory_envelopes or {},
            )
        )
        self._recurrent = build_plan_run(
            *recurrent,
            simulation=recurrent_simulation,
            bridge=bridge,
            functions=functions,
            memory_envelopes=recurrent_memory_envelopes,
        )
        if (self._initial is None) != (initial_fixed_layout is None):
            raise ValueError(
                "initial execution plan and fixed layout must be provided together"
            )
        self._initial_fixed_layout = initial_fixed_layout
        self._recurrent_fixed_layout = recurrent_fixed_layout
        self._optimizer_state_initialized = not optimizer_state_was_lazy
        self._optimizer_state_available = (
            optimizer_state_preinitialized or not optimizer_state_was_lazy
        )
        # Materialization uses a short-lived action batch. Replace it with
        # exactly one immutable initial or recurrent plan.
        self._bridge.clear_tasks()
        if self._initial is not None and not self._optimizer_state_initialized:
            assert self._initial_fixed_layout is not None
            self._initial = self._admit_run(self._initial, self._initial_fixed_layout)
            self._active_run = self._initial
        else:
            self._recurrent = self._admit_run(
                self._recurrent, self._recurrent_fixed_layout
            )
            self._active_run = self._recurrent
        self._gradients = {
            state.bridge.alias_for_object(item.gradient_object_id): model_parameter
            for item in recurrent[0].gradients
            for model_parameter in (state.model.get_parameter(item.parameter_name),)
        }
        self._invocations = 0
        self._optimizer_size_by_alias = {
            item.alias_group_id: item.size_bytes
            for item in self._recurrent.plan.program.alias_groups
        }
        self._armed_execution_timing: _ArmedExecutionTiming | None = None
        self._armed_span_timing: _ArmedSpanTiming | None = None
        self._task_annotations = TaskBoundaryAnnotations(self._bridge)
        # Detailed tracing is default-off and allocated lazily. Full-model
        # schedules emit several records per action plus readiness and
        # retirement records, so task count alone is not a safe bound. Keep a
        # deliberately generous fixed capacity and report any overflow in the
        # public reconciliation summary rather than truncating silently.
        self._trace_allocation_capacity = 1_000_000
        self._trace_event_capacity = 1_000_000
        self._trace_start_event: torch.cuda.Event | None = None
        self._trace_end_event: torch.cuda.Event | None = None
        self._trace_origin_event: torch.cuda.Event | None = None
        self._trace_task_events: dict[
            str,
            tuple[
                torch.cuda.Event,
                torch.cuda.Event,
                torch.cuda.Event,
                torch.cuda.Event,
            ],
        ] = {}
        self._span_start_event: torch.cuda.Event | None = None
        self._span_end_event: torch.cuda.Event | None = None
        self._completion = ReusableCompletionEvent(state.device)

    def _admit_run(
        self,
        run: _PlanRun,
        fixed_layout: RuntimeFixedLayout,
    ) -> _PlanRun:
        self._bridge.admit_fixed_layout(fixed_layout)
        initial_actions = tuple(
            MemoryAction("task_000000", alias_id, MemoryActionKind.FETCH)
            for alias_id in run.initial_fetches
        )
        self._bridge.admit_initial_actions(
            initial_actions,
            task_number=fixed_layout.initial_task_id,
            action_trace_labels=tuple(
                f"shadowspill.fetch.initial.{alias_id}"
                for alias_id in run.initial_fetches
            ),
        )
        labels = TransferLabelIndex(
            run.plan.program,
            {record.task.task_id: record.trace_label for record in run.execution},
        )
        admitted: list[_ExecutionTaskRecord] = []
        for record in run.execution:
            admitted.append(
                replace(
                    record,
                    task_handle=self._bridge.admit_task(
                        record.task,
                        record.input_aliases,
                        record.actions,
                        labels.labels_for(record.actions),
                        record.memory_envelope,
                        trace_label=record.trace_label,
                        publications=record.publications,
                    ),
                )
            )
        caller_aliases = tuple(
            alias_id for values in run.public_by_microbatch for alias_id in values
        )
        caller_acquisition_handle = self._bridge.admit_caller_acquisition(
            caller_aliases
        )
        self._bridge.seal_fixed_layout()
        return replace(
            run,
            execution=tuple(admitted),
            initial_task_id=fixed_layout.initial_task_id,
            caller_acquisition_handle=caller_acquisition_handle,
        )

    def prepare_execution_tracing(self) -> None:
        """Lazily allocate reusable trace buffers and timing events."""

        runs = tuple(run for run in (self._initial, self._recurrent) if run is not None)
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
        for events in self._trace_task_events.values():
            for event in events:
                event.record(stream)
        stream.synchronize()

    def __call__(
        self, inputs: Sequence[Sequence[Any]]
    ) -> tuple[tuple[torch.Tensor, ...], tuple[Any, ...]]:
        timing = self._armed_execution_timing
        run = self._begin_invocation(inputs, timing)
        self._submit_initial_placement(run, timing)
        ordered = self._execute_program(run)
        self._handoff_public_outputs(run, ordered)
        losses, metrics = self._rebuild_objective_results(ordered)
        self._invocations += 1
        if timing is not None:
            timing.dispatch_call_finished_ns = time.perf_counter_ns()
        return losses, metrics

    def _begin_invocation(
        self,
        inputs: Sequence[Sequence[Any]],
        timing: _ArmedExecutionTiming | None,
    ) -> _PlanRun:
        if timing is not None:
            timing.dispatch_call_started_ns = time.perf_counter_ns()
            timing.origin_event.record(torch.cuda.current_stream())
        if self._invocations:
            # V1 plans have a fresh terminal state. Preserve asynchronous
            # StepResult construction, but do not accidentally overlap the
            # next invocation with terminal transfers from the prior plan.
            started_ns = time.perf_counter_ns() if timing is not None else 0
            self._bridge.wait_plan_idle()
            if timing is not None:
                timing.dispatch_startup_wait_ns = time.perf_counter_ns() - started_ns
        run = (
            self._initial
            if self._initial is not None and not self._optimizer_state_initialized
            else self._recurrent
        )
        if run is None:
            raise AssertionError("initial optimizer plan is unavailable")
        if run is not self._active_run:
            self._bridge.clear_tasks()
            if run is self._initial:
                layout = self._initial_fixed_layout
                if layout is None:
                    raise AssertionError("initial fixed layout is unavailable")
                self._initial = self._admit_run(run, layout)
                run = self._initial
            else:
                self._recurrent = self._admit_run(run, self._recurrent_fixed_layout)
                run = self._recurrent
            self._active_run = run
        if timing is not None:
            self._begin_armed_runtime_trace(timing)
        self._state.refresh_inputs(inputs)
        return run

    def _begin_armed_runtime_trace(self, timing: _ArmedExecutionTiming) -> None:
        """Open the current invocation's runtime trace after prior work is idle."""

        timing.statistics_before = self._bridge.statistics()
        # Transfers are measured on their lanes from the same origin event
        # the compute-stream markers use, so every lane shares one timeline.
        self._bridge.begin_runtime_trace(
            step_id=self._invocations + 1,
            origin_event_handle=int(timing.origin_event.cuda_event),
        )

    def _submit_initial_placement(
        self,
        run: _PlanRun,
        timing: _ArmedExecutionTiming | None,
    ) -> None:
        started_ns = time.perf_counter_ns() if timing is not None else 0
        with self._profile_range("shadowspill.training.initial_actions"):
            if run.initial_task_id is None:
                raise AssertionError("run has no admitted initial-placement task")
            self._bridge.submit_initial_actions(
                tuple(
                    MemoryAction("task_000000", alias_id, MemoryActionKind.FETCH)
                    for alias_id in run.initial_fetches
                ),
                task_number=run.initial_task_id,
            )
        if timing is not None:
            timing.dispatch_initial_actions_ns = time.perf_counter_ns() - started_ns

    def _execute_program(
        self,
        run: _PlanRun,
    ) -> tuple[tuple[torch.Tensor, ...], ...]:
        public_tensors: dict[int, tuple[torch.Tensor, ...]] = {}
        for record in run.execution:
            entrypoint = record.entrypoint
            outputs = self._execute_task(run, record)
            if entrypoint.phase == "forward" and entrypoint.microbatch is not None:
                public_tensors[entrypoint.microbatch] = outputs[
                    : entrypoint.public_output_count
                ]
        return tuple(public_tensors[index] for index in range(len(public_tensors)))

    def _handoff_public_outputs(
        self,
        run: _PlanRun,
        ordered: tuple[tuple[torch.Tensor, ...], ...],
    ) -> None:
        aliases = tuple(
            alias_id for values in run.public_by_microbatch for alias_id in values
        )
        tensors = tuple(tensor for values in ordered for tensor in values)
        bindings = self._bridge.acquire_for_caller(
            aliases,
            tensors,
            acquisition_handle=run.caller_acquisition_handle,
        )
        self._bridge.transfer_outputs_to_caller(
            aliases,
            tensors,
            bindings,
            acquisition_handle=run.caller_acquisition_handle,
        )
        for alias_id in aliases:
            self._state.object_store.pop(alias_id, None)

    def _rebuild_objective_results(
        self,
        ordered: tuple[tuple[torch.Tensor, ...], ...],
    ) -> tuple[tuple[torch.Tensor, ...], tuple[Any, ...]]:
        losses: list[torch.Tensor] = []
        metrics: list[Any] = []
        for capture, values in zip(self._state.captures, ordered, strict=True):
            losses.append(values[0].detach())
            metrics.append(
                capture.objective_schema.rebuild_metrics(
                    tuple(value.detach() for value in values[1:])
                )
            )
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
            actions=(
                tuple(
                    MemoryAction("task_000000", alias_id, MemoryActionKind.FETCH)
                    for alias_id in run.initial_fetches
                )
                + run.plan.schedule.actions
            ),
            simulation=run.simulation,
            trace_setup_ns=trace_setup_ns,
            alias_accesses=_alias_accesses(run),
        )
        self._armed_execution_timing = armed

    def arm_selected_span_timing(self) -> None:
        """Arm a two-event selected-task span without detailed tracing.

        This qualification path does not enable runtime tracing, callbacks,
        profiler ranges, per-task events, allocator snapshots, or Python component
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

    def collect_step_diagnostics(self) -> StepDiagnostics:
        """Synchronize and resolve the structured trace for one real call."""

        timing = self._armed_execution_timing
        if timing is None:
            raise RuntimeError("no execution timing measurement is armed")
        try:
            return collect_step_diagnostics(timing, self._bridge)
        finally:
            self._armed_execution_timing = None

    def cancel_execution_timing(self) -> None:
        """Synchronously tear down an armed debug trace after execution failure."""

        timing = self._armed_execution_timing
        if timing is None:
            return
        stream = timing.stream or torch.cuda.current_stream()
        stream.synchronize()
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
        task.dispatch_started_ns = time.perf_counter_ns()
        task.before_task_enter_ns = task.dispatch_started_ns
        return task

    @staticmethod
    def _finish_task_timing(task: _ArmedTaskTiming | None) -> None:
        if task is None:
            return
        task.dispatch_finished_ns = time.perf_counter_ns()
        task.after_task_exit_ns = task.dispatch_finished_ns

    @staticmethod
    def _record_task_readiness(
        task: _ArmedTaskTiming | None, stream: torch.cuda.Stream | None
    ) -> None:
        if task is not None:
            if stream is None:
                raise AssertionError("task timing omitted its CUDA stream")
            task.readiness_event.record(stream)

    @staticmethod
    def _record_task_inputs_ready(
        task: _ArmedTaskTiming | None, stream: torch.cuda.Stream | None
    ) -> None:
        """Mark where waiting for inputs ends and waiting for ranges begins."""

        if task is not None:
            if stream is None:
                raise AssertionError("task timing omitted its CUDA stream")
            task.inputs_ready_event.record(stream)

    @staticmethod
    def _record_task_start(
        task: _ArmedTaskTiming | None, stream: torch.cuda.Stream | None
    ) -> None:
        if task is not None:
            if stream is None:
                raise AssertionError("task timing omitted its CUDA stream")
            task.start_event.record(stream)

    @staticmethod
    def _record_task_end(
        task: _ArmedTaskTiming | None, stream: torch.cuda.Stream | None
    ) -> None:
        if task is not None:
            if stream is None:
                raise AssertionError("task timing omitted its CUDA stream")
            task.end_event.record(stream)
            task.dispatch_after_started_ns = time.perf_counter_ns()

    def _profile_range(self, name: str) -> AbstractContextManager[None]:
        return self._task_annotations.range(name)

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
        """Synchronously snapshot optimizer state without stale CUDA pointers.

        The snapshot is independent: each tensor is its own compact host
        allocation, aliasing neither runtime storage nor the other entries,
        so a caller can serialize it while training continues. The pool keeps
        the authoritative copy throughout, and this reads it in place, so the
        snapshot is normally the only copy of the state outside the pool. An
        alias whose pool copy is not the current one is read into a buffer
        first, and that alias costs two until the snapshot is built. On a
        large model even one copy is the biggest transient the frontend asks
        for, so budget for it.
        """

        if not self._optimizer_state_initialized:
            raw = self.optimizer.state_dict()
            return {
                "state": {},
                "param_groups": copy.deepcopy(raw["param_groups"]),
            }

        exposed = self._expose_optimizer_state_cpu(self._borrowed_alias_buffer)
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
            self._restore_optimizer_spill_only(exposed)

    def load_optimizer_state(self, value: Mapping[str, object]) -> bool:
        """Restore optimizer metadata and write tensor bytes into spill storage."""

        exposed = self._expose_optimizer_state_cpu()
        initialized = False
        try:
            restored = restore_optimizer_checkpoint_structure(
                dict(self._state.model.named_parameters()),
                self.optimizer,
                value,
            )
            tensors = {item.name: item for item in restored.tensors}
            current = self._current_optimizer_bindings()
            planned = self._recurrent.lowered.optimizer_objects
            present = {item.name for item in planned if item.name in current}
            required_created = {
                item.name for item in planned if item.created_on_first_step
            }
            if present and present != {item.name for item in planned}:
                missing = sorted({item.name for item in planned} - present)
                raise RuntimeError(
                    f"optimizer checkpoint has incomplete planned state: {missing}"
                )
            initialized = not required_created or (
                restored.initialized and required_created.issubset(present)
            )
            if initialized:
                self._write_restored_optimizer_tensors(
                    planned,
                    current,
                    tensors,
                )
        finally:
            self._restore_optimizer_spill_only(exposed)

        if not initialized:
            aliases = tuple(
                self._bridge.alias_for_object(item.object_id) for item in planned
            )
            self._bridge.unregister(aliases)
            for item in planned:
                alias_id = self._bridge.alias_for_object(item.object_id)
                self._state.object_store.pop(alias_id, None)
                self._state.object_tensors.pop(item.object_id, None)
            self._optimizer_state_initialized = False
            return False

        self._optimizer_state_initialized = True
        return True

    def _write_restored_optimizer_tensors(
        self,
        planned: Sequence[Any],
        current: Mapping[str, Any],
        tensors: Mapping[str, Any],
    ) -> None:
        """Copy a checkpoint into existing spill-backed optimizer aliases."""

        for name, restored in tensors.items():
            destination = restored.destination
            source = restored.source.detach()
            if destination.device.type != "cpu":
                raise RuntimeError(
                    f"optimizer checkpoint destination {name!r} is not CPU exposed"
                )
            destination.copy_(source.to(device="cpu"))
        written: set[str] = set()
        for item in planned:
            restored = tensors.get(item.name)
            actual = current.get(item.name)
            if restored is None or actual is None:
                raise RuntimeError(
                    f"optimizer checkpoint lacks planned tensor {item.name!r}"
                )
            alias_id = self._bridge.alias_for_object(item.object_id)
            if alias_id in written:
                continue
            self._bridge.write_spill_tensor(alias_id, actual.tensor)
            written.add(alias_id)

    def release_optimizer_state(self) -> None:
        """Drop optimizer state with the plan that owns its spill storage.

        Nothing is copied: a caller who wants the state takes the checkpoint
        while the callable is open. The state tensors view storage the plan is
        about to reclaim, so they are cleared rather than left dangling -- the
        same ownership restoration a planning rollback performs. State the
        caller imported is left alone; the plan was lent it.
        """

        if not release_optimizer_state_from_plan(
            self.optimizer,
            runtime=self._state.runtime,
        ):
            return
        self.optimizer.state.clear()
        self._optimizer_state_initialized = False
        self._optimizer_state_available = False

    def _execute_task(
        self,
        run: _PlanRun,
        record: _ExecutionTaskRecord,
    ) -> tuple[torch.Tensor, ...]:
        prepared = self._before_task(run, record)
        try:
            # Do not retain the compiled result in this caller frame.  The
            # after-task boundary must be able to destroy every unadopted
            # output before it publishes actions that reuse those ranges.
            return self._after_task(prepared, self._run_compiled_task(prepared))
        except BaseException:
            self._abort_task(prepared)
            self._finish_task_timing(prepared.timing)
            raise

    def _before_task(
        self,
        run: _PlanRun,
        record: _ExecutionTaskRecord,
    ) -> _PreparedTask:
        """Acquire, rebind, and assemble one complete frontend task boundary."""

        timing = self._begin_task_timing(record.entrypoint)
        if (
            timing is None
            and not self._task_annotations.enabled
            and self._armed_span_timing is None
        ):
            return self._before_task_fast(run, record)
        runtime_scope_open = False
        try:
            with self._profile_range(f"shadowspill.before_task.{record.trace_label}"):
                stream = self._resolve_task_stream(record, timing)
                self._mark_task_readiness(timing, stream)
                with self._profile_range(
                    f"shadowspill.storage_rebind.{record.trace_label}"
                ):
                    input_tensors = self._lookup_task_inputs(record, timing)
                    self._acquire_task_inputs(record, stream, input_tensors, timing)
                    runtime_scope_open = True
                    call = self._assemble_task_call(record, timing)
                self._record_task_inputs_ready(timing, stream)
                with self._profile_range(
                    f"shadowspill.allocation_reuse.{record.trace_label}"
                ):
                    self._bridge.wait_task_allocations(
                        record.task_handle,
                        self._state.device.index or 0,
                    )
                prepared = _PreparedTask(
                    run=run,
                    record=record,
                    stream=stream,
                    arguments=call.arguments,
                    function=call.function,
                    eager_optimizer=call.eager_optimizer,
                    timing=timing,
                )
                self._record_compute_start(stream)
                self._record_task_start(timing, stream)
            if timing is not None:
                timing.dispatch_before_finished_ns = time.perf_counter_ns()
            return prepared
        except BaseException:
            if runtime_scope_open:
                self._bridge.abort_task(record.task_handle)
            self._finish_task_timing(timing)
            raise

    def _before_task_fast(
        self,
        run: _PlanRun,
        record: _ExecutionTaskRecord,
    ) -> _PreparedTask:
        """Execute the default-off-observability boundary without cold-path work."""

        runtime_scope_open = False
        try:
            # Resolve the already-admitted storage-only input vector.
            try:
                input_tensors = tuple(
                    self._state.object_store[alias_id]
                    for alias_id in record.input_storage_aliases
                )
            except KeyError as error:
                raise RuntimeError(
                    f"task input {error.args[0]!r} has no tensor binding"
                ) from error

            # Acquire readiness and install every current storage binding.
            self._bridge.before_task_and_acquire(
                record.task_handle,
                self._state.device.index or 0,
                input_tensors,
            )
            runtime_scope_open = True

            # Assemble the selected callable and its predecoded arguments.
            call = (
                self._assemble_optimizer_call(record)
                if record.entrypoint.phase == "optimizer"
                else self._assemble_graph_call(record)
            )
            return _PreparedTask(
                run=run,
                record=record,
                stream=None,
                arguments=call.arguments,
                function=call.function,
                eager_optimizer=call.eager_optimizer,
                timing=None,
            )
        except BaseException:
            if runtime_scope_open:
                self._bridge.abort_task(record.task_handle)
            raise

    def _resolve_task_stream(
        self,
        record: _ExecutionTaskRecord,
        timing: _ArmedTaskTiming | None,
    ) -> torch.cuda.Stream | None:
        needs_python_stream = timing is not None or self._armed_span_timing is not None
        return torch.cuda.current_stream() if needs_python_stream else None

    def _mark_task_readiness(
        self,
        timing: _ArmedTaskTiming | None,
        stream: torch.cuda.Stream | None,
    ) -> None:
        self._record_task_readiness(timing, stream)

    def _lookup_task_inputs(
        self,
        record: _ExecutionTaskRecord,
        timing: _ArmedTaskTiming | None,
    ) -> tuple[torch.Tensor, ...]:
        started_ns = time.perf_counter_ns() if timing is not None else 0
        tensors: list[torch.Tensor] = []
        for alias_id in record.input_storage_aliases:
            tensor = self._state.object_store.get(alias_id)
            if tensor is None:
                raise RuntimeError(f"task input {alias_id!r} has no tensor binding")
            tensors.append(tensor)
        if timing is not None:
            timing.dispatch_input_lookup_ns = time.perf_counter_ns() - started_ns
        return tuple(tensors)

    def _acquire_task_inputs(
        self,
        record: _ExecutionTaskRecord,
        stream: torch.cuda.Stream | None,
        tensors: tuple[torch.Tensor, ...],
        timing: _ArmedTaskTiming | None,
    ) -> None:
        try:
            self._acquire_input_storages(record, stream, tensors)
        except RuntimeError as error:
            states = self._bridge.input_failure_states(record.input_aliases)
            detail = "; ".join(states) if states else "all snapshots device-ready"
            raise RuntimeError(f"{error}; input_states=[{detail}]") from error

    def _acquire_input_storages(
        self,
        record: _ExecutionTaskRecord,
        stream: torch.cuda.Stream | None,
        tensors: tuple[torch.Tensor, ...],
    ) -> None:
        if record.task_handle == 0:
            raise AssertionError("execution task has no admitted handle")
        self._bridge.before_task_and_acquire(
            record.task_handle,
            self._state.device.index or 0,
            tensors,
        )

    def _assemble_task_call(
        self,
        record: _ExecutionTaskRecord,
        timing: _ArmedTaskTiming | None,
    ) -> _TaskCall:
        started_ns = time.perf_counter_ns() if timing is not None else 0
        if record.entrypoint.phase == "optimizer":
            result = self._assemble_optimizer_call(record)
        else:
            result = self._assemble_graph_call(record)
        if timing is not None:
            timing.dispatch_argument_assembly_ns = time.perf_counter_ns() - started_ns
        return result

    def _assemble_optimizer_call(self, record: _ExecutionTaskRecord) -> _TaskCall:
        artifact = record.entrypoint.artifact
        eager = isinstance(artifact, OpaqueOptimizerArtifact) or (
            not self._optimizer_state_available
        )
        if eager:
            return _TaskCall((), None, True)
        if not isinstance(artifact, GraphArtifact):
            raise RuntimeError("optimizer task has no executable artifact")
        object_ids = record.optimizer_argument_object_ids
        if all(object_id is not None for object_id in object_ids):
            try:
                arguments = tuple(
                    self._state.object_tensors[object_id]
                    for object_id in object_ids
                    if object_id is not None
                )
            except KeyError as error:
                raise RuntimeError(
                    f"optimizer object {error.args[0]!r} is unbound"
                ) from error
        else:
            current = self._current_optimizer_bindings()
            try:
                arguments = tuple(
                    current[name].tensor
                    for name in record.entrypoint.optimizer_binding_names
                )
            except KeyError as error:
                raise RuntimeError(
                    f"optimizer tensor {error.args[0]!r} is unbound"
                ) from error
        return _TaskCall(arguments, record.function, False)

    def _assemble_graph_call(self, record: _ExecutionTaskRecord) -> _TaskCall:
        if not isinstance(record.entrypoint.artifact, GraphArtifact):
            raise RuntimeError("graph task has no captured artifact")
        if record.argument_template is None:
            raise AssertionError("graph argument template is absent")
        arguments = list(record.argument_template)
        for slot in record.entrypoint.input_slots:
            arguments[slot.leaf_index] = self._state.object_tensors[slot.object_id]
        return _TaskCall(arguments, record.function, False)

    def _run_compiled_task(self, prepared: _PreparedTask) -> object:
        """Dispatch only the numerical task represented by ``prepared``."""

        started_ns = time.perf_counter_ns() if prepared.timing is not None else 0
        if prepared.timing is not None:
            prepared.timing.before_task_exit_ns = started_ns
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
            invoked_ns = time.perf_counter_ns()
            prepared.timing.dispatch_invoke_ns = invoked_ns - started_ns
            prepared.timing.after_task_enter_ns = invoked_ns
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

        annotation_id = (
            self._task_annotations.begin(
                f"shadowspill.after_task.{prepared.record.trace_label}"
            )
            if self._task_annotations.enabled
            else 0
        )
        try:
            processed, dematerialized = self._prepare_task_publication(
                prepared, raw_outputs
            )
            # The compiled result tuple owns every unadopted task output.  Drop
            # this outer reference before publishing the task boundary so its
            # allocator frees become causal predecessors of any action that
            # reuses the task's spatial ranges.
            del raw_outputs
            self._publish_task_to_runtime(prepared, processed, dematerialized)
            self._publish_frontend_bindings(prepared, processed)
            if (
                prepared.record.released_ephemeral
                or prepared.record.task.task_id
                == prepared.run.lowered.optimizer_task_id
            ):
                self._finish_task_cleanup(prepared)
            outputs = processed.outputs
        finally:
            if annotation_id:
                self._task_annotations.end(annotation_id)
        self._finish_task_timing(prepared.timing)
        return outputs

    def _prepare_task_publication(
        self,
        prepared: _PreparedTask,
        raw_outputs: object,
    ) -> tuple[_ProcessedTaskOutputs, tuple[torch.Tensor, ...]]:
        annotation_id = (
            self._task_annotations.begin(
                f"shadowspill.output_processing.{prepared.record.trace_label}"
            )
            if self._task_annotations.enabled
            else 0
        )
        try:
            processed = self._process_task_outputs(prepared, raw_outputs)
            dematerialized = self._dematerialization_tensors(
                prepared.record,
                processed.adopted,
            )
        finally:
            if annotation_id:
                self._task_annotations.end(annotation_id)
        return processed, dematerialized

    def _publish_task_to_runtime(
        self,
        prepared: _PreparedTask,
        processed: _ProcessedTaskOutputs,
        dematerialized: tuple[torch.Tensor, ...],
    ) -> None:
        annotation_id = (
            self._task_annotations.begin(
                f"shadowspill.runtime.after_task.{prepared.record.trace_label}"
            )
            if self._task_annotations.enabled
            else 0
        )
        try:
            self._publish_admitted_task(prepared, processed, dematerialized)
        finally:
            if annotation_id:
                self._task_annotations.end(annotation_id)
        prepared.runtime_scope_open = False

    def _publish_admitted_task(
        self,
        prepared: _PreparedTask,
        processed: _ProcessedTaskOutputs,
        dematerialized: tuple[torch.Tensor, ...],
    ) -> None:
        record = prepared.record
        if record.task_handle == 0:
            raise AssertionError("execution task has no admitted handle")
        try:
            self._bridge.after_task_and_update(
                record.task_handle,
                self._state.device.index or 0,
                processed.adopted,
                tuple(item.publication_ordinal for item in processed.adopted),
                dematerialized,
                replacements=processed.replacements,
            )
        except RuntimeError as error:
            diagnostics = read_allocator_failure(
                self._bridge.library,
                "after_task storage publication",
                task=record.identity,
            )
            if diagnostics is not None:
                if diagnostics.is_allocator_oom:
                    raise allocator_oom_error(diagnostics) from error
                raise generic_runtime_error(diagnostics) from error
            raise RuntimeError(
                "after_task storage publication failed for "
                f"execution_{record.execution_ordinal:06d} "
                f"({record.semantic_name}): {error}"
            ) from error

    def _publish_frontend_bindings(
        self,
        prepared: _PreparedTask,
        processed: _ProcessedTaskOutputs,
    ) -> None:
        started_ns = time.perf_counter_ns() if prepared.timing is not None else 0
        replacement_by_alias = (
            {item.alias_id: item for item in processed.replacements}
            if processed.replacements
            else {}
        )
        replacement_aliases = (
            processed.replacement_aliases if processed.replacements else ()
        )
        for publication in processed.adopted:
            alias_id = publication.alias_id
            if alias_id in replacement_aliases:
                self._state.publish_replacement_views(replacement_by_alias[alias_id])
            else:
                self._state.object_store[alias_id] = publication.tensor
        if processed.optimizer_bindings:
            for object_id, tensor, alias_id in processed.optimizer_bindings:
                self._state.object_store.setdefault(alias_id, tensor)
                self._state.object_tensors[object_id] = tensor
            self._optimizer_state_available = True
        if prepared.timing is not None:
            prepared.timing.dispatch_output_state_publish_ns = (
                time.perf_counter_ns() - started_ns
            )

    def _finish_task_cleanup(self, prepared: _PreparedTask) -> None:
        started_ns = time.perf_counter_ns() if prepared.timing is not None else 0
        with self._profile_range(f"shadowspill.cleanup.{prepared.record.trace_label}"):
            self._cleanup_after_task(prepared)
        if prepared.timing is not None:
            prepared.timing.dispatch_cleanup_ns = time.perf_counter_ns() - started_ns

    def _process_task_outputs(
        self,
        prepared: _PreparedTask,
        raw_outputs: object,
    ) -> _ProcessedTaskOutputs:
        outputs: tuple[torch.Tensor, ...] = ()
        adopted: tuple[PublishedStorage, ...] = ()
        replacement_aliases: frozenset[str] = frozenset()
        optimizer_bindings: tuple[tuple[str, torch.Tensor, str], ...] = ()
        entrypoint = prepared.record.entrypoint
        timing = prepared.timing
        if entrypoint.phase == "optimizer":
            started_ns = time.perf_counter_ns() if timing is not None else 0
            if prepared.eager_optimizer and not self._optimizer_state_available:
                adopted, optimizer_bindings = self._created_optimizer_state(
                    prepared.record
                )
            else:
                optimizer_bindings = ()
            if timing is not None:
                timing.dispatch_output_publish_ns = time.perf_counter_ns() - started_ns
        else:
            started_ns = time.perf_counter_ns() if timing is not None else 0
            if isinstance(raw_outputs, (tuple, list)):
                leaves = raw_outputs
            else:
                leaves, _ = tree_flatten(raw_outputs)
            if timing is not None:
                timing.dispatch_output_flatten_ns = time.perf_counter_ns() - started_ns
            started_ns = time.perf_counter_ns() if timing is not None else 0
            if entrypoint.phase == "forward":
                if not all(isinstance(value, torch.Tensor) for value in leaves):
                    raise RuntimeError("captured forward graph returned a static leaf")
                tensor_outputs = tuple(cast(torch.Tensor, value) for value in leaves)
                adopted, replacement_aliases = self._bind_forward_outputs(
                    prepared.record,
                    tensor_outputs,
                    timing,
                )
                if entrypoint.public_output_leaves:
                    outputs = tuple(
                        tensor_outputs[index]
                        for index in entrypoint.public_output_leaves
                    )
            else:
                adopted = self._accumulate_gradients(prepared.record, leaves, timing)
            if timing is not None:
                timing.dispatch_output_publish_ns = time.perf_counter_ns() - started_ns
            del leaves
        replacements = (
            tuple(
                self._state.replacement_storage_views(alias_id)
                for item in adopted
                for alias_id in (item.alias_id,)
                if alias_id in replacement_aliases
            )
            if replacement_aliases
            else ()
        )
        return _ProcessedTaskOutputs(
            outputs,
            adopted,
            replacements,
            optimizer_bindings,
        )

    def _cleanup_after_task(self, prepared: _PreparedTask) -> None:
        self._forget_released_objects(prepared.run, prepared.record)
        if prepared.record.task.task_id == prepared.run.lowered.optimizer_task_id:
            self._optimizer_state_initialized = True
            for parameter in self._gradients.values():
                parameter.grad = None
            for alias_id in self._gradients:
                self._state.object_store.pop(alias_id, None)
            for gradient_binding in prepared.run.lowered.gradients:
                self._state.object_tensors.pop(
                    gradient_binding.gradient_object_id, None
                )

    def _abort_task(
        self,
        prepared: _PreparedTask,
    ) -> None:
        if prepared.runtime_scope_open:
            prepared.runtime_scope_open = False
            self._bridge.abort_task(
                prepared.record.task_handle,
            )

    def _bind_forward_outputs(
        self,
        record: _ExecutionTaskRecord,
        outputs: tuple[torch.Tensor, ...],
        timing: _ArmedTaskTiming | None,
    ) -> tuple[tuple[PublishedStorage, ...], frozenset[str]]:
        started_ns = time.perf_counter_ns() if timing is not None else 0
        adopted: list[PublishedStorage] = []
        replacements: set[str] = set()
        for item in record.forward_outputs:
            tensor = outputs[item.leaf_index]
            if item.adopt and item.publication_ordinal is not None:
                adopted.append(
                    PublishedStorage(
                        tensor,
                        item.alias_id,
                        item.publication_ordinal,
                    )
                )
            if item.replace:
                replacements.add(item.alias_id)
            else:
                self._state.object_store.setdefault(item.alias_id, tensor)
                self._state.object_tensors[item.object_id] = tensor
        if timing is not None:
            timing.dispatch_output_classification_ns = (
                time.perf_counter_ns() - started_ns
            )
        return tuple(adopted), frozenset(replacements)

    def _accumulate_gradients(
        self,
        record: _ExecutionTaskRecord,
        leaves: Sequence[object],
        timing: _ArmedTaskTiming | None,
    ) -> tuple[PublishedStorage, ...]:
        started_ns = time.perf_counter_ns() if timing is not None else 0
        contributions: list[torch.Tensor] = []
        destinations: list[torch.Tensor] = []
        first: list[tuple[str, str, torch.Tensor, int | None]] = []
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
                first.append(
                    (
                        item.object_id,
                        item.alias_id,
                        contribution,
                        item.publication_ordinal,
                    )
                )
            elif _same_tensor_view(destination, contribution):
                self._state.object_tensors[item.object_id] = destination
                parameter = self._gradients.get(item.alias_id)
                if parameter is not None:
                    parameter.grad = destination
            else:
                destinations.append(destination)
                contributions.append(contribution)
        if timing is not None:
            timing.dispatch_output_classification_ns = (
                time.perf_counter_ns() - started_ns
            )
        adopted: list[PublishedStorage] = []
        for _object_id, alias_id, contribution, publication_ordinal in first:
            if publication_ordinal is not None:
                adopted.append(
                    PublishedStorage(
                        contribution,
                        alias_id,
                        publication_ordinal,
                    )
                )
        started_ns = time.perf_counter_ns() if timing is not None else 0
        for object_id, alias_id, contribution, _publication_ordinal in first:
            self._state.object_store[alias_id] = contribution
            self._state.object_tensors[object_id] = contribution
            parameter = self._gradients.get(alias_id)
            if parameter is not None:
                parameter.grad = contribution
        if timing is not None:
            timing.dispatch_output_state_publish_ns = (
                time.perf_counter_ns() - started_ns
            )
        if destinations:
            torch._foreach_add_(destinations, contributions)
        return tuple(adopted)

    def _created_optimizer_state(
        self,
        record: _ExecutionTaskRecord,
    ) -> tuple[
        tuple[PublishedStorage, ...],
        tuple[tuple[str, torch.Tensor, str], ...],
    ]:
        artifact = record.entrypoint.artifact
        if not isinstance(artifact, OpaqueOptimizerArtifact):
            raise RuntimeError("initial optimizer state requires an opaque artifact")
        outputs = {
            binding.name: binding.tensor
            for binding in opaque_optimizer_outputs(
                artifact,
                self.optimizer,
                device_ordinal=self._state.device.index or 0,
            )
        }
        produced: set[str] = set()
        adopted: list[PublishedStorage] = []
        bound: list[tuple[str, torch.Tensor, str]] = []
        for item in record.optimizer_outputs:
            tensor = outputs.get(item.name)
            if tensor is None:
                raise RuntimeError(
                    f"optimizer did not create planned state {item.name!r}"
                )
            if item.alias_id not in produced and item.publication_ordinal is not None:
                adopted.append(
                    PublishedStorage(
                        tensor,
                        item.alias_id,
                        item.publication_ordinal,
                    )
                )
                produced.add(item.alias_id)
            bound.append((item.object_id, tensor, item.alias_id))
        return tuple(adopted), tuple(bound)

    def _current_optimizer_bindings(self) -> dict[str, Any]:
        return {
            item.name: item
            for item in current_optimizer_bindings(
                dict(self._state.model.named_parameters()), self.optimizer
            )
        }

    def _copied_alias_buffer(self, alias_id: str) -> torch.Tensor:
        """Return a writable host buffer holding one alias's current bytes."""

        owner = torch.empty(
            self._optimizer_size_by_alias[alias_id],
            dtype=torch.uint8,
            device="cpu",
        )
        self._bridge.read_spill_tensor(alias_id, owner)
        return owner

    def _borrowed_alias_buffer(self, alias_id: str) -> torch.Tensor:
        """Return one alias's bytes in place, copying only if it must."""

        window = self._bridge.spill_window(alias_id)
        return self._copied_alias_buffer(alias_id) if window is None else window

    def _expose_optimizer_state_cpu(
        self,
        owner_for: Callable[[str], torch.Tensor] | None = None,
    ) -> tuple[_ExposedOptimizerTensor, ...]:
        """Point live optimizer state at host bytes, however they are obtained.

        ``owner_for`` supplies each alias group's bytes. The default copies
        them out of the pool into a writable buffer, which a caller that
        intends to write back must use; a read-only caller can pass
        ``_borrowed_alias_buffer`` to read the pool in place instead.
        """

        make_owner = self._copied_alias_buffer if owner_for is None else owner_for
        self._bridge.wait_plan_idle()
        current = self._current_optimizer_bindings()
        exposed: list[_ExposedOptimizerTensor] = []
        owners: dict[str, torch.Tensor] = {}
        for item in self._recurrent.lowered.optimizer_objects:
            actual = current.get(item.name)
            if actual is None:
                continue
            tensor = actual.tensor
            alias_id = self._bridge.alias_for_object(item.object_id)
            owner = owners.get(alias_id)
            if owner is None:
                owner = make_owner(alias_id)
                owners[alias_id] = owner
            device_placeholder = tensor.data
            layout = _TensorLayout(
                tuple(tensor.shape),
                tuple(tensor.stride()),
                int(tensor.storage_offset()),
                tensor.dtype,
            )
            tensor.data = self._view(owner, layout)
            exposed.append(_ExposedOptimizerTensor(tensor, device_placeholder))
        return tuple(exposed)

    def _restore_optimizer_spill_only(
        self, exposed: tuple[_ExposedOptimizerTensor, ...]
    ) -> None:
        # Exposing state never changes the neutral runtime object. Restore the
        # exact dematerialized CUDA views that were present before the CPU
        # snapshot; manufacturing temporary device allocations here would add
        # no information and can exceed the execution pool for large AdamW
        # inventories even though every individual task is feasible.
        for item in exposed:
            item.tensor.data = item.device_placeholder

    @staticmethod
    def _view(owner: torch.Tensor, layout: _TensorLayout) -> torch.Tensor:
        return torch.empty(0, dtype=layout.dtype, device=owner.device).set_(
            owner.untyped_storage(),
            layout.storage_offset,
            layout.shape,
            layout.stride,
        )

    def _dematerialization_tensors(
        self,
        record: _ExecutionTaskRecord,
        adopted: tuple[PublishedStorage, ...],
    ) -> tuple[torch.Tensor, ...]:
        if not record.dematerialize_aliases:
            return ()
        newly_produced = {item.alias_id: item.tensor for item in adopted}
        pending: list[torch.Tensor] = []
        for alias_id in record.dematerialize_aliases:
            tensor = newly_produced.get(alias_id)
            if tensor is None:
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
            for object_id in object_ids:
                self._state.object_tensors.pop(object_id, None)


def _same_tensor_view(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Return whether two tensors name the same bytes with the same geometry."""

    return bool(
        left.untyped_storage()._cdata == right.untyped_storage()._cdata
        and left.storage_offset() == right.storage_offset()
        and left.shape == right.shape
        and left.stride() == right.stride()
        and left.dtype == right.dtype
    )


__all__ = ["TrainingExecutor"]
