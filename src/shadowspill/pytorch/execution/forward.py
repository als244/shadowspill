"""Ordinary PyTorch task dispatch wrapped by exact runtime boundaries."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils._pytree import TreeSpec, tree_flatten, tree_unflatten

from shadowspill.ir import ExecutionPlan, MemoryAction, MemoryActionKind, TaskSpec
from shadowspill.pytorch.lowering.forward import LoweredForwardProgram, TaskEntrypoint
from shadowspill.pytorch.materialization.forward import MaterializedForwardState
from shadowspill.pytorch.partition import PartitionedExport
from shadowspill.pytorch.runtime_adapter.abi import ObjectBinding
from shadowspill.pytorch.runtime_adapter.bridge import (
    RuntimeBridge,
    TaskMemoryEnvelope,
    actions_by_task,
)
from shadowspill.pytorch.runtime_adapter.failures import ExecutionTaskIdentity
from shadowspill.pytorch.runtime_adapter.fixed_layout import RuntimeFixedLayout
from shadowspill.pytorch.runtime_adapter.transfer_labels import TransferLabelIndex


@dataclass(slots=True)
class _PreparedForwardTask:
    arguments: tuple[object, ...]
    input_aliases: tuple[str, ...]
    stream: torch.cuda.Stream
    runtime_scope_open: bool = True


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
    ) -> None:
        super().__init__()
        self._entrypoint = entrypoint
        self._task = task
        self._function = function
        self._bridge = bridge
        self._state = state
        self._actions = actions
        self._identity = identity

    def forward(self, *arguments: object) -> object:
        prepared = self._before_task(arguments)
        try:
            raw_outputs = self._run_compiled_task(prepared)
            return self._after_task(prepared, raw_outputs)
        except BaseException:
            self._abort_task(prepared)
            raise

    def _before_task(self, arguments: tuple[object, ...]) -> _PreparedForwardTask:
        leaves, _ = tree_flatten(arguments)
        input_aliases = tuple(
            self._bridge.alias_for_object(slot.object_id)
            for slot in self._entrypoint.input_slots
        )
        stream = torch.cuda.current_stream()
        runtime_scope_open = False
        try:
            bindings = self._bridge.before_task(
                self._task.task_id, stream, input_aliases
            )
            runtime_scope_open = True
            rebound: list[tuple[torch.Tensor, str, ObjectBinding]] = []
            compiled_arguments: list[torch.Tensor] = []
            for slot, alias_id, binding in zip(
                self._entrypoint.input_slots,
                input_aliases,
                bindings,
                strict=True,
            ):
                tensor = leaves[slot.leaf_index]
                if not isinstance(tensor, torch.Tensor):
                    raise RuntimeError("task tensor input became static")
                rebound.append((tensor, alias_id, binding))
                compiled_arguments.append(tensor)
            self._bridge.rebind_many(rebound)
            for tensor, alias_id, binding in rebound:
                self._state.object_store[alias_id] = tensor
                self._state.generations[alias_id] = binding.generation
            return _PreparedForwardTask(
                tuple(compiled_arguments), input_aliases, stream
            )
        except BaseException:
            if runtime_scope_open:
                self._bridge.abort_task()
            raise

    def _run_compiled_task(self, prepared: _PreparedForwardTask) -> object:
        # Forward-only execution has no captured backward. Avoid creating
        # hidden dispatcher-autograd contexts across planned task bounds.
        with torch.no_grad():
            return self._function(*prepared.arguments)

    def _after_task(
        self,
        prepared: _PreparedForwardTask,
        output: object,
    ) -> object:
        output_leaves, _ = tree_flatten(output)
        produced: set[str] = set()
        rebound: list[tuple[torch.Tensor, str, ObjectBinding]] = []
        replacement_leaves = set(self._entrypoint.replacement_output_leaves)
        for slot in self._entrypoint.output_slots:
            tensor = output_leaves[slot.leaf_index]
            if not isinstance(tensor, torch.Tensor):
                raise RuntimeError("task tensor output became static")
            alias_id = self._bridge.alias_for_object(slot.object_id)
            replacement = slot.leaf_index in replacement_leaves
            if replacement and alias_id not in produced:
                binding = self._bridge.replace_output(alias_id, tensor)
                self._state.replace_alias_generation(
                    alias_id, tensor, binding.generation
                )
                produced.add(alias_id)
            elif alias_id not in prepared.input_aliases and alias_id not in produced:
                binding = self._bridge.promote_output(alias_id, tensor)
                rebound.append((tensor, alias_id, binding))
                produced.add(alias_id)
                self._state.object_store[alias_id] = tensor
            elif not replacement:
                self._state.object_store[alias_id] = tensor
        self._bridge.rebind_many(rebound)
        for _, alias_id, binding in rebound:
            self._state.generations[alias_id] = binding.generation
        dematerialized: list[tuple[torch.Tensor, str, int]] = []
        handoff_sources = {
            self._bridge.alias_for_object(item.source_object_id)
            for item in self._entrypoint.storage_handoffs
            if item.destination_object_id in self._task.outputs
        }
        for action in self._actions:
            if action.kind not in {
                MemoryActionKind.RELEASE,
                MemoryActionKind.OFFLOAD,
            }:
                continue
            alias_id = action.alias_group_id
            if alias_id in handoff_sources:
                continue
            tensor = self._state.object_store.get(alias_id)
            generation = self._state.generations.get(alias_id)
            if tensor is None or generation is None:
                raise RuntimeError(
                    f"action references unbound alias group {alias_id!r}"
                )
            dematerialized.append((tensor, alias_id, generation))
        self._bridge.dematerialize_many(dematerialized)
        self._bridge.after_task(
            self._task.task_id,
            prepared.stream,
            self._task.mutations,
            self._actions,
        )
        prepared.runtime_scope_open = False
        return output

    def _abort_task(
        self,
        prepared: _PreparedForwardTask,
    ) -> None:
        if prepared.runtime_scope_open:
            prepared.runtime_scope_open = False
            self._bridge.abort_task()


class ForwardExecutor:
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
        task_by_id = {task.task_id: task for task in plan.program.tasks}
        grouped_actions = actions_by_task(plan.schedule.actions)
        self._initial_prefetches = tuple(
            item.alias_group_id
            for item in plan.schedule.initial_residency
            if item.location.value == "device"
            and bridge.requires_storage(item.alias_group_id)
        )
        initial_actions = tuple(
            self._initial_prefetch_action(alias_id)
            for alias_id in self._initial_prefetches
        )
        # Materialization uses short-lived legacy action records.  They are
        # idle now and must not become part of the immutable execution plan.
        bridge.clear_execution_plan()
        bridge.admit_fixed_layout(fixed_layout)
        bridge.admit_initial_actions(
            initial_actions,
            task_number=fixed_layout.initial_task_id,
            action_trace_labels=tuple(
                f"shadowspill.fetch.initial.{alias_id}"
                for alias_id in self._initial_prefetches
            ),
        )
        trace_labels = {
            entrypoint.task_id: (
                f"execution_{execution_ordinal:06d}.forward."
                f"stage_{execution_ordinal:04d}.{entrypoint.module_target}"
            )
            for execution_ordinal, entrypoint in enumerate(lowered.entrypoints)
        }
        bridge.configure_task_labels(trace_labels)
        transfer_labels = TransferLabelIndex(plan.program, trace_labels)
        for execution_ordinal, entrypoint in enumerate(lowered.entrypoints):
            task = task_by_id[entrypoint.task_id]
            task_actions = grouped_actions.get(entrypoint.task_id, ())
            bridge.admit_execution(
                task,
                tuple(
                    bridge.alias_for_object(slot.object_id)
                    for slot in entrypoint.input_slots
                ),
                task_actions,
                transfer_labels.labels_for(task_actions),
                memory_envelopes.get(task.task_id, TaskMemoryEnvelope()),
            )
            function = functions[entrypoint.artifact.compatibility_digest]
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
                        f"{entrypoint.module_target}"
                    ),
                    canonical_task_id=task.task_id,
                ),
            )
            self._root.set_submodule(entrypoint.module_target, wrapper)
        bridge.seal_fixed_layout()
        output_objects = {
            item.object_id
            for item in plan.program.objects
            if item.role.value == "output"
        }
        self._caller_output_aliases = tuple(
            dict.fromkeys(
                bridge.alias_for_object(slot.object_id)
                for entrypoint in lowered.entrypoints
                for slot in entrypoint.output_slots
                if slot.object_id in output_objects
            )
        )
        self._initial_task_id = fixed_layout.initial_task_id
        self._invocations = 0
        self._profiler_annotations_enabled = False

    def set_profiler_annotations(self, enabled: bool) -> None:
        """Toggle provider annotations for this forward callable."""

        self._bridge.set_profiler_annotations(enabled)
        self._profiler_annotations_enabled = enabled

    def finish_profiler_annotations(self) -> None:
        """Drain annotated asynchronous work before disabling its provider."""

        if not self._profiler_annotations_enabled:
            return
        self._bridge.wait_idle()
        self.set_profiler_annotations(False)

    def __call__(self, arguments: Sequence[object]) -> object:
        if self._invocations:
            # Forward v1 is also non-cyclic: begin only after the preceding
            # invocation reaches its declared terminal residency.
            self._bridge.wait_idle()
        root_arguments = self._state.refresh_inputs(arguments)
        initial_actions = tuple(
            self._initial_prefetch_action(alias_id)
            for alias_id in self._initial_prefetches
        )
        self._bridge.submit_initial_actions(
            initial_actions,
            task_number=self._initial_task_id,
        )
        flat_output = self._root(*root_arguments)
        output_leaves, _ = tree_flatten(flat_output)
        output = tree_unflatten(
            [output_leaves[index] for index in self._user_output_indices],
            self._output_tree_spec,
        )
        caller_tensors = tuple(
            self._state.object_store[alias_id]
            for alias_id in self._caller_output_aliases
        )
        bindings = self._bridge.acquire_for_caller(
            self._caller_output_aliases,
            caller_tensors,
            task_number=(1 << 59) + self._invocations,
        )
        self._bridge.transfer_outputs_to_caller(
            self._caller_output_aliases, caller_tensors, bindings
        )
        for alias_id in self._caller_output_aliases:
            self._state.object_store.pop(alias_id, None)
            self._state.generations.pop(alias_id, None)
        self._invocations += 1
        return output

    @staticmethod
    def _initial_prefetch_action(alias_id: str) -> MemoryAction:
        return MemoryAction("task_000000", alias_id, MemoryActionKind.PREFETCH)


__all__ = ["ForwardExecutor"]
