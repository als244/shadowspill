"""Exact accumulated-training dispatch through selected AOT graph pairs."""

from __future__ import annotations

import copy
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from typing import Any, cast

import torch
from torch.utils._pytree import tree_flatten, tree_map

from shadowspill.ir import ExecutionPlan, MemoryAction, MemoryActionKind
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.diagnostics.collection import collect_step_diagnostics
from shadowspill.pytorch.diagnostics.execution import (
    ExecutionTiming,
    StepDiagnostics,
    TaskExecutionTiming,
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
from shadowspill.pytorch.lowering.training import (
    LoweredTrainingProgram,
    TrainingTaskEntrypoint,
)
from shadowspill.pytorch.materialization.training import TrainingMaterializedState
from shadowspill.pytorch.optimizer import (
    OpaqueOptimizerArtifact,
    current_optimizer_bindings,
    opaque_optimizer_outputs,
    restore_optimizer_checkpoint_structure,
)
from shadowspill.pytorch.runtime_adapter.bridge import (
    RuntimeBridge,
    TaskMemoryEnvelope,
)
from shadowspill.pytorch.runtime_adapter.fixed_layout import RuntimeFixedLayout
from shadowspill.pytorch.runtime_adapter.transfer_labels import TransferLabelIndex
from shadowspill.pytorch.state.optimizer import release_optimizer_state_from_plan

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
    cuda_placeholder: torch.Tensor


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
    adopted: tuple[tuple[torch.Tensor, str], ...]
    replacement_aliases: frozenset[str]
    optimizer_bindings: tuple[tuple[str, torch.Tensor, str], ...] = ()


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
        self._initial = (
            None
            if initial is None
            else build_plan_run(
                *initial,
                bridge=bridge,
                functions=functions,
                memory_envelopes=initial_memory_envelopes or {},
            )
        )
        self._recurrent = build_plan_run(
            *recurrent,
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
        # Materialization uses short-lived legacy action records.  Replace
        # them with exactly one immutable initial or recurrent plan.
        self._bridge.clear_execution_plan()
        if self._initial is not None and not self._optimizer_state_initialized:
            assert self._initial_fixed_layout is not None
            self._initial = self._admit_run(self._initial, self._initial_fixed_layout)
            self._active_run = self._initial
        else:
            self._recurrent = self._admit_run(
                self._recurrent, self._recurrent_fixed_layout
            )
            self._active_run = self._recurrent
        self._trace_label_run: _PlanRun | None = None
        self._configure_task_trace_labels(self._active_run)
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

    def _admit_run(
        self,
        run: _PlanRun,
        fixed_layout: RuntimeFixedLayout,
    ) -> _PlanRun:
        self._bridge.admit_fixed_layout(fixed_layout)
        initial_actions = tuple(
            MemoryAction("task_000000", alias_id, MemoryActionKind.PREFETCH)
            for alias_id in run.initial_prefetches
        )
        self._bridge.admit_initial_actions(
            initial_actions,
            task_number=fixed_layout.initial_task_id,
            action_trace_labels=tuple(
                f"shadowspill.fetch.initial.{alias_id}"
                for alias_id in run.initial_prefetches
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
                    native_handle=self._bridge.admit_execution(
                        record.task,
                        record.input_aliases,
                        record.actions,
                        labels.labels_for(record.actions),
                        record.memory_envelope,
                    ),
                )
            )
        self._bridge.seal_fixed_layout()
        return replace(
            run,
            execution=tuple(admitted),
            initial_task_id=fixed_layout.initial_task_id,
        )

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
        run = self._begin_invocation(inputs, timing)
        self._submit_initial_placement(run, timing)
        ordered = self._execute_program(run)
        self._handoff_public_outputs(run, ordered)
        losses, metrics = self._rebuild_objective_results(ordered)
        self._invocations += 1
        if timing is not None:
            timing.host_call_finished_ns = time.perf_counter_ns()
        return losses, metrics

    def _begin_invocation(
        self,
        inputs: Sequence[Sequence[Any]],
        timing: _ArmedExecutionTiming | None,
    ) -> _PlanRun:
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
        if run is not self._active_run:
            self._bridge.clear_execution_plan()
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
        if run is not self._trace_label_run:
            self._configure_task_trace_labels(run)
        self._state.refresh_inputs(inputs)
        return run

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
                    MemoryAction("task_000000", alias_id, MemoryActionKind.PREFETCH)
                    for alias_id in run.initial_prefetches
                ),
                task_number=run.initial_task_id,
            )
        if timing is not None:
            timing.host_initial_actions_ns = time.perf_counter_ns() - started_ns

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
            task_number=(1 << 59) + self._invocations,
        )
        self._bridge.transfer_outputs_to_caller(aliases, tensors, bindings)
        for alias_id in aliases:
            self._state.object_store.pop(alias_id, None)
            self._state.generations.pop(alias_id, None)

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
            statistics_before=self._bridge.statistics(),
            actions=(
                tuple(
                    MemoryAction("task_000000", alias_id, MemoryActionKind.PREFETCH)
                    for alias_id in run.initial_prefetches
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
            self._restore_optimizer_host_only(exposed)

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

    def restore_optimizer_cpu(self) -> None:
        """Leave live optimizer state backed by ordinary CPU storage."""

        self._expose_optimizer_state_cpu()
        release_optimizer_state_from_plan(
            self.optimizer,
            runtime=self._state.runtime,
        )

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

        timing = self._begin_task_timing(record.entrypoint)
        runtime_scope_open = False
        try:
            with self._profile_range(f"shadowspill.before_task.{record.trace_label}"):
                stream = self._resolve_task_stream(record, timing)
                self._mark_task_readiness(timing, stream)
                rebind_started_ns = time.perf_counter_ns() if timing is not None else 0
                with self._profile_range(
                    f"shadowspill.storage_rebind.{record.trace_label}"
                ):
                    input_tensors = self._lookup_task_inputs(record, timing)
                    generations = self._acquire_task_inputs(
                        record, stream, input_tensors, timing
                    )
                    runtime_scope_open = True
                    self._publish_input_generations(record, generations, timing)
                    call = self._assemble_task_call(record, timing)
                if timing is not None:
                    timing.host_rebind_ns = time.perf_counter_ns() - rebind_started_ns
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
            return prepared
        except BaseException as error:
            if runtime_scope_open:
                self._bridge.abort_task_after_failure(
                    f"prepare task {record.task.task_id}",
                    error,
                    task=record.identity,
                )
            self._finish_task_timing(timing)
            raise

    def _resolve_task_stream(
        self,
        record: _ExecutionTaskRecord,
        timing: _ArmedTaskTiming | None,
    ) -> torch.cuda.Stream | None:
        started_ns = time.perf_counter_ns() if timing is not None else 0
        needs_python_stream = (
            timing is not None
            or self._armed_span_timing is not None
            or not record.native_handle
        )
        stream = torch.cuda.current_stream() if needs_python_stream else None
        if timing is not None:
            timing.host_stream_resolution_ns = time.perf_counter_ns() - started_ns
        return stream

    def _mark_task_readiness(
        self,
        timing: _ArmedTaskTiming | None,
        stream: torch.cuda.Stream | None,
    ) -> None:
        started_ns = time.perf_counter_ns() if timing is not None else 0
        self._record_task_readiness(timing, stream)
        if timing is not None:
            timing.host_readiness_marker_ns = time.perf_counter_ns() - started_ns

    def _lookup_task_inputs(
        self,
        record: _ExecutionTaskRecord,
        timing: _ArmedTaskTiming | None,
    ) -> tuple[torch.Tensor, ...]:
        started_ns = time.perf_counter_ns() if timing is not None else 0
        tensors: list[torch.Tensor] = []
        for alias_id in record.input_aliases:
            tensor = self._state.object_store.get(alias_id)
            if tensor is None:
                raise RuntimeError(f"task input {alias_id!r} has no tensor binding")
            tensors.append(tensor)
        if timing is not None:
            timing.host_input_lookup_ns = time.perf_counter_ns() - started_ns
        return tuple(tensors)

    def _acquire_task_inputs(
        self,
        record: _ExecutionTaskRecord,
        stream: torch.cuda.Stream | None,
        tensors: tuple[torch.Tensor, ...],
        timing: _ArmedTaskTiming | None,
    ) -> tuple[int, ...]:
        started_ns = time.perf_counter_ns() if timing is not None else 0
        try:
            generations = self._acquire_input_generations(record, stream, tensors)
        except RuntimeError as error:
            states = self._bridge.input_failure_states(record.input_aliases)
            detail = "; ".join(states) if states else "all snapshots device-ready"
            raise RuntimeError(f"{error}; input_states=[{detail}]") from error
        if timing is not None:
            timing.host_native_before_task_ns = time.perf_counter_ns() - started_ns
        return generations

    def _acquire_input_generations(
        self,
        record: _ExecutionTaskRecord,
        stream: torch.cuda.Stream | None,
        tensors: tuple[torch.Tensor, ...],
    ) -> tuple[int, ...]:
        if record.native_handle:
            return self._bridge.before_execution_and_acquire(
                record.native_handle,
                record.dense_task_id,
                self._state.device.index or 0,
                tensors,
                record.input_aliases,
            )
        bindings = self._bridge.before_task(
            record.task.task_id,
            stream if stream is not None else torch.cuda.current_stream(),
            record.input_aliases,
        )
        self._bridge.rebind_many(
            tuple(zip(tensors, record.input_aliases, bindings, strict=True))
        )
        return tuple(binding.generation for binding in bindings)

    def _publish_input_generations(
        self,
        record: _ExecutionTaskRecord,
        generations: tuple[int, ...],
        timing: _ArmedTaskTiming | None,
    ) -> None:
        started_ns = time.perf_counter_ns() if timing is not None else 0
        self._state.generations.update(
            zip(record.input_aliases, generations, strict=True)
        )
        if timing is not None:
            timing.host_generation_publish_ns = time.perf_counter_ns() - started_ns

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
            timing.host_argument_assembly_ns = time.perf_counter_ns() - started_ns
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

        with self._profile_range(
            f"shadowspill.after_task.{prepared.record.trace_label}"
        ):
            processed, dematerialized = self._prepare_task_publication(
                prepared, raw_outputs
            )
            # The compiled result tuple owns every unadopted task output.  Drop
            # this outer reference before publishing the task boundary so its
            # allocator frees become causal predecessors of any action that
            # reuses the task's spatial ranges.
            del raw_outputs
            generations = self._publish_task_to_runtime(
                prepared, processed, dematerialized
            )
            self._publish_output_generations(prepared, processed, generations)
            self._finish_task_cleanup(prepared)
            return processed.outputs

    def _prepare_task_publication(
        self,
        prepared: _PreparedTask,
        raw_outputs: object,
    ) -> tuple[_ProcessedTaskOutputs, tuple[torch.Tensor, ...]]:
        started_ns = time.perf_counter_ns() if prepared.timing is not None else 0
        with self._profile_range(
            f"shadowspill.output_processing.{prepared.record.trace_label}"
        ):
            processed = self._process_task_outputs(prepared, raw_outputs)
            dematerialized = self._dematerialization_tensors(
                prepared.record,
                processed.adopted,
            )
        if prepared.timing is not None:
            prepared.timing.host_postprocess_ns = time.perf_counter_ns() - started_ns
        return processed, dematerialized

    def _publish_task_to_runtime(
        self,
        prepared: _PreparedTask,
        processed: _ProcessedTaskOutputs,
        dematerialized: tuple[torch.Tensor, ...],
    ) -> tuple[int, ...]:
        started_ns = time.perf_counter_ns() if prepared.timing is not None else 0
        with self._profile_range(
            f"shadowspill.runtime.after_task.{prepared.record.trace_label}"
        ):
            if prepared.record.native_handle:
                generations = self._publish_native_task(
                    prepared, processed, dematerialized
                )
            else:
                generations = self._publish_legacy_task(
                    prepared, processed, dematerialized
                )
        prepared.runtime_scope_open = False
        if prepared.timing is not None:
            prepared.timing.host_native_after_task_ns = (
                time.perf_counter_ns() - started_ns
            )
        return generations

    def _publish_native_task(
        self,
        prepared: _PreparedTask,
        processed: _ProcessedTaskOutputs,
        dematerialized: tuple[torch.Tensor, ...],
    ) -> tuple[int, ...]:
        record = prepared.record
        try:
            return self._bridge.after_execution_and_update(
                record.native_handle,
                record.dense_task_id,
                self._state.device.index or 0,
                processed.adopted,
                dematerialized,
                replacement_aliases=processed.replacement_aliases,
            )
        except RuntimeError as error:
            raise RuntimeError(
                "after_task storage publication failed for "
                f"execution_{record.execution_ordinal:06d} "
                f"({record.semantic_name}): {error}"
            ) from error

    def _publish_legacy_task(
        self,
        prepared: _PreparedTask,
        processed: _ProcessedTaskOutputs,
        dematerialized: tuple[torch.Tensor, ...],
    ) -> tuple[int, ...]:
        record = prepared.record
        generations = self._bridge.adopt_many(
            processed.adopted,
            replacement_aliases=processed.replacement_aliases,
        )
        pending = self._legacy_dematerialization_bindings(
            record,
            dematerialized,
            processed.adopted,
            generations,
        )
        self._bridge.dematerialize_many(pending)
        self._bridge.after_task(
            record.task.task_id,
            prepared.stream
            if prepared.stream is not None
            else torch.cuda.current_stream(),
            record.task.mutations,
            record.actions,
        )
        return generations

    def _legacy_dematerialization_bindings(
        self,
        record: _ExecutionTaskRecord,
        tensors: tuple[torch.Tensor, ...],
        adopted: tuple[tuple[torch.Tensor, str], ...],
        adopted_generations: tuple[int, ...],
    ) -> tuple[tuple[torch.Tensor, str, int], ...]:
        new_generations = {
            alias_id: generation
            for (_tensor, alias_id), generation in zip(
                adopted,
                adopted_generations,
                strict=True,
            )
        }
        pending: list[tuple[torch.Tensor, str, int]] = []
        for tensor, alias_id in zip(tensors, record.dematerialize_aliases, strict=True):
            generation = new_generations.get(alias_id)
            if generation is None:
                generation = self._state.generations.get(alias_id)
            if generation is None:
                raise RuntimeError(f"action references unbound generation {alias_id!r}")
            pending.append((tensor, alias_id, generation))
        return tuple(pending)

    def _publish_output_generations(
        self,
        prepared: _PreparedTask,
        processed: _ProcessedTaskOutputs,
        generations: tuple[int, ...],
    ) -> None:
        started_ns = time.perf_counter_ns() if prepared.timing is not None else 0
        for (tensor, alias_id), generation in zip(
            processed.adopted, generations, strict=True
        ):
            if alias_id in processed.replacement_aliases:
                self._state.replace_alias_generation(alias_id, tensor, generation)
            else:
                self._state.generations[alias_id] = generation
        if processed.optimizer_bindings:
            generation_by_alias = dict(
                zip(
                    (alias_id for _tensor, alias_id in processed.adopted),
                    generations,
                    strict=True,
                )
            )
            for object_id, tensor, alias_id in processed.optimizer_bindings:
                self._state.object_store.setdefault(alias_id, tensor)
                self._state.object_tensors[object_id] = tensor
                self._state.generations[alias_id] = generation_by_alias[alias_id]
            self._optimizer_state_available = True
        if prepared.timing is not None:
            prepared.timing.host_output_state_publish_ns = (
                time.perf_counter_ns() - started_ns
            )

    def _finish_task_cleanup(self, prepared: _PreparedTask) -> None:
        started_ns = time.perf_counter_ns() if prepared.timing is not None else 0
        with self._profile_range(f"shadowspill.cleanup.{prepared.record.trace_label}"):
            self._cleanup_after_task(prepared)
        if prepared.timing is not None:
            prepared.timing.host_cleanup_ns = time.perf_counter_ns() - started_ns

    def _process_task_outputs(
        self,
        prepared: _PreparedTask,
        raw_outputs: object,
    ) -> _ProcessedTaskOutputs:
        outputs: tuple[torch.Tensor, ...] = ()
        adopted: tuple[tuple[torch.Tensor, str], ...] = ()
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
                    tensor_outputs[index] for index in entrypoint.public_output_leaves
                )
            else:
                adopted = self._accumulate_gradients(prepared.record, leaves, timing)
            if timing is not None:
                timing.host_output_publish_ns = time.perf_counter_ns() - started_ns
            del leaves
        return _ProcessedTaskOutputs(
            outputs,
            adopted,
            replacement_aliases,
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
                f"execute task {prepared.record.task.task_id}",
                error,
                task=prepared.record.identity,
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

    def _created_optimizer_state(
        self,
        record: _ExecutionTaskRecord,
    ) -> tuple[
        tuple[tuple[torch.Tensor, str], ...],
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
        adopted: list[tuple[torch.Tensor, str]] = []
        bound: list[tuple[str, torch.Tensor, str]] = []
        for item in record.optimizer_outputs:
            tensor = outputs.get(item.name)
            if tensor is None:
                raise RuntimeError(
                    f"optimizer did not create planned state {item.name!r}"
                )
            if item.alias_id not in produced:
                adopted.append((tensor, item.alias_id))
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

    def _expose_optimizer_state_cpu(
        self,
    ) -> tuple[_ExposedOptimizerTensor, ...]:
        self._bridge.wait_idle()
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
                owner = torch.empty(
                    self._optimizer_size_by_alias[alias_id],
                    dtype=torch.uint8,
                    device="cpu",
                )
                self._bridge.read_spill_tensor(alias_id, owner)
                owners[alias_id] = owner
            cuda_placeholder = tensor.data
            layout = _TensorLayout(
                tuple(tensor.shape),
                tuple(tensor.stride()),
                int(tensor.storage_offset()),
                tensor.dtype,
            )
            tensor.data = self._view(owner, layout)
            exposed.append(_ExposedOptimizerTensor(tensor, cuda_placeholder))
        return tuple(exposed)

    def _restore_optimizer_host_only(
        self, exposed: tuple[_ExposedOptimizerTensor, ...]
    ) -> None:
        # Exposing state never changes the neutral runtime object. Restore the
        # exact dematerialized CUDA views that were present before the CPU
        # snapshot; manufacturing temporary device allocations here would add
        # no information and can exceed the execution pool for large AdamW
        # inventories even though every individual task is feasible.
        for item in exposed:
            item.tensor.data = item.cuda_placeholder

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
        adopted: tuple[tuple[torch.Tensor, str], ...],
    ) -> tuple[torch.Tensor, ...]:
        newly_produced = {alias_id: tensor for tensor, alias_id in adopted}
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
            self._state.generations.pop(alias_id, None)
            for object_id in object_ids:
                self._state.object_tensors.pop(object_id, None)

    def _configure_task_trace_labels(self, run: _PlanRun) -> dict[str, str]:
        """Register labels for the one execution program selected next."""

        result = {record.task.task_id: record.trace_label for record in run.execution}
        self._bridge.configure_task_labels(result)
        self._trace_label_run = run
        return result


def _same_tensor_view(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Return whether two tensors name the same bytes with the same geometry."""

    return bool(
        left.untyped_storage()._cdata == right.untyped_storage()._cdata
        and left.storage_offset() == right.storage_offset()
        and left.shape == right.shape
        and left.stride() == right.stride()
        and left.dtype == right.dtype
    )


__all__ = ["ExecutionTiming", "TaskExecutionTiming", "TrainingExecutor"]
