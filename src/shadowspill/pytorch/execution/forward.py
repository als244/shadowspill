"""Ordinary PyTorch task dispatch wrapped by exact runtime boundaries."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils._pytree import TreeSpec, tree_flatten, tree_unflatten

from shadowspill.diagnostics.timing import ArmedTaskTiming as _ArmedTaskTiming
from shadowspill.errors import PlanningError
from shadowspill.ir import ExecutionPlan, MemoryAction, MemoryActionKind, TaskSpec
from shadowspill.ir.schedule import first_use_initial_order
from shadowspill.pytorch.invocation import ReusableCompletionEvent
from shadowspill.pytorch.lowering.forward import LoweredForwardProgram, TaskEntrypoint
from shadowspill.pytorch.materialization.forward import MaterializedForwardState
from shadowspill.pytorch.materialization.replacement import ReplacementStorageViews
from shadowspill.pytorch.partition import PartitionedExport
from shadowspill.pytorch.runtime_adapter.boundaries import (
    PublishedStorage,
    acquire_for_caller,
    after_task_and_update,
    before_task_and_acquire,
    submit_initial_actions,
    transfer_outputs_to_caller,
)
from shadowspill.pytorch.sharing import ResolvedSharedOutput, TensorRef, format_path
from shadowspill.runtime.failures import ExecutionTaskIdentity
from shadowspill.runtime.fixed_layout import RuntimeFixedLayout
from shadowspill.runtime.plan import (
    RuntimeBridge,
    TaskMemoryEnvelope,
    TaskPublication,
    abort_task,
    actions_by_task,
    admit_caller_acquisition,
    admit_fixed_layout,
    admit_initial_actions,
    admit_task,
    clear_tasks,
    seal_fixed_layout,
)
from shadowspill.runtime.transfer_labels import TransferLabelIndex
from shadowspill.simulator import SimulationResult

from .annotations import AnnotatedExecutor, TaskBoundaryAnnotations
from .timing import (
    ExecutionTiming,
    TracedInvocation,
    TracedTask,
    alias_accesses,
)


@dataclass(slots=True)
class _PreparedForwardTask:
    arguments: tuple[object, ...]
    input_aliases: tuple[str, ...]
    #: Resolved only when something records on it -- a trace, or the
    #: always-on span this task opens or closes. Otherwise the boundary
    #: never asks the framework which stream it is on.
    stream: torch.cuda.Stream | None = None
    runtime_scope_open: bool = True


@dataclass(frozen=True, slots=True)
class _ProcessedForwardOutputs:
    raw: object
    adopted: tuple[PublishedStorage, ...]
    replacements: tuple[ReplacementStorageViews, ...]
    replacement_aliases: frozenset[str]
    bindings: tuple[tuple[str, torch.Tensor], ...]
    dematerialized: tuple[tuple[str, torch.Tensor], ...]


def _forward_publications(
    entrypoint: TaskEntrypoint,
    input_aliases: tuple[str, ...],
    bridge: RuntimeBridge,
) -> tuple[TaskPublication, ...]:
    """Predecode the unique storage roots this task can publish."""

    produced: set[str] = set()
    result: list[TaskPublication] = []
    replacement_leaves = set(entrypoint.replacement_output_leaves)
    for slot in entrypoint.output_slots:
        alias_id = bridge.objects.alias_for_object(slot.object_id)
        replace_lease = slot.leaf_index in replacement_leaves
        adopt = (replace_lease or alias_id not in input_aliases) and (
            alias_id not in produced
        )
        if not adopt:
            continue
        produced.add(alias_id)
        if bridge.objects.requires_storage(alias_id):
            result.append(TaskPublication(alias_id, replace_lease))
    return tuple(result)


class _ExecutingStage(nn.Module):
    def __init__(
        self,
        entrypoint: TaskEntrypoint,
        task: TaskSpec,
        function: Callable[..., object],
        bridge: RuntimeBridge,
        state: MaterializedForwardState,
        actions: tuple[MemoryAction, ...],
        identity: ExecutionTaskIdentity,
        task_handle: int,
        publications: tuple[TaskPublication, ...],
        annotations: TaskBoundaryAnnotations,
        timing: ExecutionTiming,
        ends_compute_span: bool,
    ) -> None:
        super().__init__()
        self._timing = timing
        self._ends_compute_span = ends_compute_span
        self._entrypoint = entrypoint
        self._task = task
        self._function = function
        self._bridge = bridge
        self._state = state
        self._actions = actions
        self._identity = identity
        self._task_handle = task_handle
        self._annotations = annotations
        self._trace_label = f"{identity.execution_task_id}.{identity.semantic_name}"
        self._publication_ordinals = {
            item.alias_id: ordinal for ordinal, item in enumerate(publications)
        }
        self._device_ordinal = state.device.index or 0
        self._input_aliases = tuple(
            bridge.objects.alias_for_object(slot.object_id)
            for slot in entrypoint.input_slots
        )
        self._input_storage_indices = tuple(
            index
            for index, alias_id in enumerate(self._input_aliases)
            if bridge.objects.requires_storage(alias_id)
        )

    def forward(self, *arguments: object) -> object:
        task = self._timing.begin_task(self._entrypoint)
        prepared = self._before_task(arguments, task)
        try:
            raw_outputs = self._run_compiled_task(prepared, task)
            return self._after_task(prepared, raw_outputs, task)
        except BaseException:
            self._abort_task(prepared)
            self._timing.finish_task(task)
            raise

    def _before_task(
        self,
        arguments: tuple[object, ...],
        task: _ArmedTaskTiming | None,
    ) -> _PreparedForwardTask:
        runtime_scope_open = False
        stream = self._task_stream(task)
        try:
            with self._annotations.range(
                f"shadowspill.before_task.{self._trace_label}"
            ):
                self._timing.record_task_readiness(task, stream)
                input_tensors = self._resolve_inputs(arguments)
                acquire_started_ns = time.perf_counter_ns() if task else 0
                before_task_and_acquire(
                    self._bridge,
                    self._task_handle,
                    self._device_ordinal,
                    tuple(
                        input_tensors[index] for index in self._input_storage_indices
                    ),
                )
                if task is not None:
                    task.dispatch_input_acquire_ns = (
                        time.perf_counter_ns() - acquire_started_ns
                    )
                runtime_scope_open = True
                self._publish_input_bindings(input_tensors)
                self._timing.record_task_inputs_ready(task, stream)
                prepared = _PreparedForwardTask(
                    input_tensors, self._input_aliases, stream
                )
            self._timing.record_compute_start(stream)
            self._timing.record_task_start(task, stream)
            return prepared
        except BaseException:
            if runtime_scope_open:
                abort_task(self._bridge, self._task_handle)
            self._timing.finish_task(task)
            raise

    def _task_stream(self, task: _ArmedTaskTiming | None) -> torch.cuda.Stream | None:
        """The stream to record on, asked for only when something records."""

        if (
            task is None
            and not self._timing.span_pending
            and (not self._ends_compute_span)
        ):
            return None
        return torch.cuda.current_stream()

    def _resolve_inputs(
        self, arguments: tuple[object, ...]
    ) -> tuple[torch.Tensor, ...]:
        leaves, _ = tree_flatten(arguments)
        tensors: list[torch.Tensor] = []
        for slot in self._entrypoint.input_slots:
            tensor = leaves[slot.leaf_index]
            if not isinstance(tensor, torch.Tensor):
                raise RuntimeError("task tensor input became static")
            tensors.append(tensor)
        return tuple(tensors)

    def _publish_input_bindings(
        self,
        tensors: tuple[torch.Tensor, ...],
    ) -> None:
        for tensor, alias_id in zip(tensors, self._input_aliases, strict=True):
            self._state.object_store[alias_id] = tensor

    def _run_compiled_task(
        self,
        prepared: _PreparedForwardTask,
        task: _ArmedTaskTiming | None,
    ) -> object:
        # Forward-only execution has no captured backward. Avoid creating
        # hidden dispatcher-autograd problems across planned task bounds.
        if task is not None:
            task.before_task_exit_ns = time.perf_counter_ns()
        with (
            self._annotations.range(f"shadowspill.compiled_call.{self._trace_label}"),
            torch.no_grad(),
        ):
            outputs = self._function(*prepared.arguments)
        if task is not None:
            task.after_task_enter_ns = time.perf_counter_ns()
        self._timing.record_task_end(task, prepared.stream)
        if self._ends_compute_span:
            self._timing.record_compute_end(prepared.stream)
        return outputs

    def _after_task(
        self,
        prepared: _PreparedForwardTask,
        output: object,
        task: _ArmedTaskTiming | None,
    ) -> object:
        with self._annotations.range(f"shadowspill.after_task.{self._trace_label}"):
            processed = self._process_outputs(output)
            after_task_and_update(
                self._bridge,
                self._task_handle,
                self._device_ordinal,
                processed.adopted,
                tuple(range(len(processed.adopted))),
                tuple(tensor for _, tensor in processed.dematerialized),
                replacements=processed.replacements,
            )
            prepared.runtime_scope_open = False
            self._publish_output_bindings(processed)
            self._forget_released_bindings(processed)
            result = processed.raw
        self._timing.finish_task(task)
        return result

    def _process_outputs(self, output: object) -> _ProcessedForwardOutputs:
        output_leaves, _ = tree_flatten(output)
        produced: set[str] = set()
        adopted: list[PublishedStorage] = []
        replacement_aliases: set[str] = set()
        bindings: list[tuple[str, torch.Tensor]] = []
        replacement_leaves = set(self._entrypoint.replacement_output_leaves)
        for slot in self._entrypoint.output_slots:
            tensor = output_leaves[slot.leaf_index]
            if not isinstance(tensor, torch.Tensor):
                raise RuntimeError("task tensor output became static")
            alias_id = self._bridge.objects.alias_for_object(slot.object_id)
            replacement = slot.leaf_index in replacement_leaves
            if replacement and alias_id not in produced:
                adopted.append(
                    PublishedStorage(
                        tensor,
                        alias_id,
                        self._publication_ordinals.get(alias_id, -1),
                    )
                )
                replacement_aliases.add(alias_id)
                produced.add(alias_id)
            elif alias_id not in self._input_aliases and alias_id not in produced:
                adopted.append(
                    PublishedStorage(
                        tensor,
                        alias_id,
                        self._publication_ordinals.get(alias_id, -1),
                    )
                )
                produced.add(alias_id)
            bindings.append((alias_id, tensor))
        replacements = tuple(
            self._state.replacement_storage_views(alias_id)
            for item in adopted
            for alias_id in (item.alias_id,)
            if alias_id in replacement_aliases
        )
        # Existing frontend views must win for an overwritten object. Backend
        # publication rebinds that stable view to the successor generation,
        # then dematerializes it if the plan immediately releases or evicts
        # the object. The compiled replacement tensor is only the temporary
        # source lease; dematerializing it would leave the stable view naming
        # a retired address on the next invocation.
        available = dict(bindings)
        available.update(self._state.object_store)
        dematerialized: list[tuple[str, torch.Tensor]] = []
        adopted_aliases = {item.alias_id for item in adopted}
        handoff_sources = {
            self._bridge.objects.alias_for_object(item.source_object_id)
            for item in self._entrypoint.storage_handoffs
            if item.destination_object_id in self._task.outputs
        }
        for action in self._actions:
            if action.kind not in {
                MemoryActionKind.RELEASE,
                MemoryActionKind.EVICT,
            }:
                continue
            alias_id = action.alias_group_id
            if alias_id in handoff_sources:
                continue
            tensor = available.get(alias_id)
            if tensor is None or (
                alias_id not in self._state.object_store
                and alias_id not in adopted_aliases
            ):
                raise RuntimeError(
                    f"action references unbound alias group {alias_id!r}"
                )
            dematerialized.append((alias_id, tensor))
        return _ProcessedForwardOutputs(
            raw=output,
            adopted=tuple(adopted),
            replacements=replacements,
            replacement_aliases=frozenset(replacement_aliases),
            bindings=tuple(bindings),
            dematerialized=tuple(dematerialized),
        )

    def _publish_output_bindings(
        self,
        processed: _ProcessedForwardOutputs,
    ) -> None:
        replacement_by_alias = {item.alias_id: item for item in processed.replacements}
        for alias_id, tensor in processed.bindings:
            self._state.object_store[alias_id] = tensor
        for item in processed.adopted:
            tensor = item.tensor
            alias_id = item.alias_id
            if alias_id in processed.replacement_aliases:
                self._state.publish_replacement_views(replacement_by_alias[alias_id])
            else:
                self._state.object_store[alias_id] = tensor

    def _forget_released_bindings(self, processed: _ProcessedForwardOutputs) -> None:
        adopted = {item.alias_id for item in processed.adopted}
        for alias_id, _ in processed.dematerialized:
            if alias_id in adopted:
                continue
            self._state.object_store.pop(alias_id, None)

    def _abort_task(
        self,
        prepared: _PreparedForwardTask,
    ) -> None:
        if prepared.runtime_scope_open:
            prepared.runtime_scope_open = False
            abort_task(self._bridge, self._task_handle)


class ForwardExecutor(AnnotatedExecutor):
    """Execute one selected forward plan and return ordinary output tensors."""

    def __init__(
        self,
        partitioned: PartitionedExport,
        lowered: LoweredForwardProgram,
        plan: ExecutionPlan,
        bridge: RuntimeBridge,
        state: MaterializedForwardState,
        functions: dict[str, Callable[..., object]],
        user_output_indices: tuple[int, ...],
        output_tree_spec: TreeSpec,
        *,
        simulation: SimulationResult,
        shared_outputs: tuple[ResolvedSharedOutput, ...] = (),
        fixed_layout: RuntimeFixedLayout,
        memory_envelopes: Mapping[str, TaskMemoryEnvelope],
    ) -> None:
        self._root = partitioned.root
        self._lowered = lowered
        self._plan = plan
        self._bridge = bridge
        self._state = state
        self._user_output_indices = user_output_indices
        self._output_tree_spec = output_tree_spec
        self._shared_outputs = tuple(shared_outputs)
        self._task_annotations = TaskBoundaryAnnotations(bridge)
        self.timing = ExecutionTiming(
            bridge, tuple(item.task_id for item in lowered.entrypoints)
        )
        task_by_id = {task.task_id: task for task in plan.program.tasks}
        grouped_actions = actions_by_task(plan.schedule.actions)
        self._initial_fetches = tuple(
            alias_group_id
            for alias_group_id in first_use_initial_order(plan.program, plan.schedule)
            if bridge.objects.requires_storage(alias_group_id)
        )
        initial_actions = tuple(
            self._initial_fetch_action(alias_id) for alias_id in self._initial_fetches
        )
        # Materialization uses a short-lived action batch. It is idle now and
        # must not become part of the immutable execution plan.
        clear_tasks(bridge)
        admit_fixed_layout(bridge, fixed_layout)
        admit_initial_actions(
            bridge,
            initial_actions,
            task_number=fixed_layout.initial_task_id,
            action_trace_labels=tuple(
                f"shadowspill.fetch.initial.{alias_id}"
                for alias_id in self._initial_fetches
            ),
        )
        trace_labels = {
            entrypoint.task_id: (
                f"execution_{execution_ordinal:06d}.forward."
                f"stage_{execution_ordinal:04d}.{entrypoint.options.target}"
            )
            for execution_ordinal, entrypoint in enumerate(lowered.entrypoints)
        }
        transfer_labels = TransferLabelIndex(plan.program, trace_labels)
        for execution_ordinal, entrypoint in enumerate(lowered.entrypoints):
            task = task_by_id[entrypoint.task_id]
            task_actions = grouped_actions.get(entrypoint.task_id, ())
            input_aliases = tuple(
                bridge.objects.alias_for_object(slot.object_id)
                for slot in entrypoint.input_slots
            )
            publications = _forward_publications(entrypoint, input_aliases, bridge)
            task_handle = admit_task(
                bridge,
                task,
                input_aliases,
                task_actions,
                transfer_labels.labels_for(task_actions),
                memory_envelopes.get(task.task_id, TaskMemoryEnvelope()),
                trace_label=trace_labels[entrypoint.task_id],
                publications=publications,
            )
            artifact = lowered.executables[entrypoint.task_id]
            function = functions[artifact.compatibility_digest]
            wrapper = _ExecutingStage(
                entrypoint,
                task,
                function,
                bridge,
                state,
                task_actions,
                ExecutionTaskIdentity(
                    execution_task_id=f"execution_{execution_ordinal:06d}",
                    semantic_name=(
                        f"forward.stage_{execution_ordinal:04d}."
                        f"{entrypoint.options.target}"
                    ),
                    canonical_task_id=task.task_id,
                ),
                task_handle,
                publications,
                self._task_annotations,
                self.timing,
                execution_ordinal == len(lowered.entrypoints) - 1,
            )
            self._root.set_submodule(
                entrypoint.options.target or entrypoint.task_id, wrapper
            )
        seal_fixed_layout(bridge)
        self._traced = self._describe_for_tracing(plan, task_by_id, simulation)
        self._public_output_aliases = tuple(
            bridge.objects.alias_for_object(object_id)
            for object_id in lowered.public_outputs
        )
        shared_indices = {item.public_leaf_index for item in self._shared_outputs}
        shared_aliases = {
            self._public_output_aliases[index] for index in shared_indices
        }
        partially_shared = {
            alias_id
            for index, alias_id in enumerate(self._public_output_aliases)
            if alias_id in shared_aliases and index not in shared_indices
        }
        if partially_shared:
            raise PlanningError(
                "all public views of one shared storage root must be declared "
                f"together: aliases={sorted(partially_shared)}"
            )
        self._caller_output_aliases = tuple(
            dict.fromkeys(
                alias_id
                for index, alias_id in enumerate(self._public_output_aliases)
                if index not in shared_indices
            )
        )
        self._caller_acquisition_handle = admit_caller_acquisition(
            bridge, self._caller_output_aliases
        )
        self._active_shared_outputs: dict[int, TensorRef] = {}
        self._completion = ReusableCompletionEvent(
            bridge.runtime._runtime_handle, state.device
        )
        self._initial_task_id = fixed_layout.initial_task_id
        self._invocations = 0

    def _describe_for_tracing(
        self,
        plan: ExecutionPlan,
        task_by_id: Mapping[str, TaskSpec],
        simulation: SimulationResult,
    ) -> TracedInvocation:
        """This program in the terms a runtime trace is taken in."""

        profiles = {item.profile_id: item for item in plan.program.profiles}
        entrypoints = tuple(enumerate(self._lowered.entrypoints))
        return TracedInvocation(
            tasks=tuple(
                TracedTask(
                    entrypoint,
                    profiles[task_by_id[entrypoint.task_id].profile_id].runtime_ns
                    / 1e9,
                    execution_ordinal,
                    f"forward.stage_{execution_ordinal:04d}."
                    f"{entrypoint.options.target}",
                )
                for execution_ordinal, entrypoint in entrypoints
            ),
            actions=(
                tuple(
                    self._initial_fetch_action(alias_id)
                    for alias_id in self._initial_fetches
                )
                + plan.schedule.actions
            ),
            simulation=simulation,
            alias_accesses=alias_accesses(
                plan.program,
                (
                    (execution_ordinal, task_by_id[entrypoint.task_id])
                    for execution_ordinal, entrypoint in entrypoints
                ),
            ),
        )

    def release_timing(self) -> None:
        """Give every marker this executor holds back to the runtime."""

        self._completion.release()
        self.timing.release()

    def arm_runtime_trace(self, *, trace_setup_ns: int = 0) -> None:
        """Bracket the next invocation, which is then traced in full."""

        self.timing.arm(self._traced, trace_setup_ns=trace_setup_ns)

    def __call__(self, arguments: Sequence[object]) -> object:
        timing = self.timing.armed
        if timing is not None:
            timing.dispatch_call_started_ns = time.perf_counter_ns()
        stream = torch.cuda.current_stream()
        self.timing.record_origin(stream)
        timing_timeline = self.timing.begin_invocation(self._invocations + 1, stream)
        if timing is not None:
            timing.timeline = timing_timeline
        self.timing.prior_invocation_drain_ns = 0
        if self._invocations:
            # Forward v1 is also non-cyclic: begin only after the preceding
            # invocation reaches its declared terminal residency.
            #
            # Timed on every invocation rather than only a traced one: the
            # first invocation has nothing to wait for, so a trace taken on a
            # warm first call is exactly the call that never pays this.
            started_ns = time.perf_counter_ns()
            self._bridge.wait_until_idle()
            self.timing.prior_invocation_drain_ns = time.perf_counter_ns() - started_ns
            if timing is not None:
                timing.prior_invocation_drain_ns = self.timing.prior_invocation_drain_ns
            self._release_closed_shared_output_generations()
        if timing is not None:
            self.timing.begin_armed_runtime_trace(timing, self._invocations + 1)
        # Staging the inputs waits for the whole runtime, so every earlier
        # call has drained here and must have left the layout empty.
        root_arguments = self._state.refresh_inputs(arguments)
        self._bridge.require_empty_layout()
        initial_actions = tuple(
            self._initial_fetch_action(alias_id) for alias_id in self._initial_fetches
        )
        started_ns = time.perf_counter_ns() if timing is not None else 0
        submit_initial_actions(
            self._bridge,
            initial_actions,
            task_number=self._initial_task_id,
        )
        if timing is not None:
            timing.dispatch_initial_actions_ns = time.perf_counter_ns() - started_ns
        flat_output = self._root(*root_arguments)
        output_leaves, _ = tree_flatten(flat_output)
        public_leaves = [output_leaves[index] for index in self._user_output_indices]
        caller_tensors = tuple(
            self._state.object_store[alias_id]
            for alias_id in self._caller_output_aliases
        )
        if self._caller_output_aliases:
            bindings = acquire_for_caller(
                self._bridge,
                self._caller_output_aliases,
                caller_tensors,
                acquisition_handle=self._caller_acquisition_handle,
            )
            transfer_outputs_to_caller(
                self._bridge,
                self._caller_output_aliases,
                caller_tensors,
                bindings,
                acquisition_handle=self._caller_acquisition_handle,
            )
        created = self._retain_shared_outputs(public_leaves)
        for alias_id in dict.fromkeys(self._public_output_aliases):
            self._state.object_store.pop(alias_id, None)
        self._invocations += 1
        self._active_shared_outputs = created
        if timing is not None:
            timing.dispatch_call_finished_ns = time.perf_counter_ns()
        return tree_unflatten(public_leaves, self._output_tree_spec)

    def validate_invocation(self) -> None:
        """Reject a call that would overwrite a still-owned output slot."""

        self._require_shared_output_slots_available()

    def prepare_invocation(self, arguments: Sequence[object]) -> Sequence[object]:
        """Resolve public shared references to the callable's tensor shells."""

        return self._state.prepare_invocation(arguments)

    def _require_shared_output_slots_available(self) -> None:
        busy = []
        for item in self._shared_outputs:
            reference = self._active_shared_outputs.get(item.public_leaf_index)
            if reference is not None and not reference.closed:
                busy.append(item)
        if busy:
            paths = ", ".join(format_path(item.path) for item in busy)
            raise RuntimeError(
                "shared output slots remain owned by the preceding invocation; "
                f"close these references before calling again: {paths}"
            )

    def _retain_shared_outputs(
        self,
        public_leaves: list[object],
    ) -> dict[int, TensorRef]:
        created: dict[int, TensorRef] = {}
        try:
            for output in self._shared_outputs:
                tensor = public_leaves[output.public_leaf_index]
                if not isinstance(tensor, torch.Tensor):
                    raise RuntimeError(
                        f"shared output {format_path(output.path)} became non-tensor"
                    )
                alias_id = self._public_output_aliases[output.public_leaf_index]
                object_reference = self._bridge.objects.acquire_object_reference(
                    alias_id
                )
                try:
                    generation = self._bridge.objects.current_generation(alias_id)
                    reference = TensorRef.from_tensor(
                        object_reference,
                        tensor,
                        generation=generation,
                        retained_pools=output.retain_in,
                    )
                except BaseException:
                    object_reference.close()
                    raise
                created[output.public_leaf_index] = reference
                public_leaves[output.public_leaf_index] = reference
            return created
        except BaseException:
            for reference in created.values():
                reference.close()
            raise

    def _release_closed_shared_output_generations(self) -> None:
        released: dict[str, int] = {}
        for index, reference in self._active_shared_outputs.items():
            if not reference.closed:
                raise RuntimeError("shared output ownership changed after validation")
            alias_id = self._public_output_aliases[index]
            previous = released.get(alias_id)
            if previous is not None:
                if previous != reference.generation:
                    raise RuntimeError(
                        "shared views of one object reference different generations"
                    )
                continue
            self._bridge.objects.release_object_generation(
                alias_id,
                expected_generation=reference.generation,
            )
            released[alias_id] = reference.generation
        self._active_shared_outputs.clear()

    @staticmethod
    def _initial_fetch_action(alias_id: str) -> MemoryAction:
        return MemoryAction("task_000000", alias_id, MemoryActionKind.FETCH)


__all__ = ["ForwardExecutor"]
