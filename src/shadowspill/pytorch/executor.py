"""Ordinary PyTorch task dispatch wrapped by exact runtime boundaries."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
import torch.nn as nn
from torch.utils._pytree import TreeSpec, tree_flatten, tree_unflatten

from shadowspill.ir import ExecutionPlan, MemoryAction, MemoryActionKind, TaskSpec

from .lowering import LoweredForwardProgram, TaskEntrypoint
from .materialization import MaterializedForwardState
from .partition import PartitionedExport
from .runtime_bridge import RuntimeBridge, actions_by_task


class _ExecutingStage(nn.Module):
    def __init__(
        self,
        entrypoint: TaskEntrypoint,
        task: TaskSpec,
        function: Callable[..., object],
        bridge: RuntimeBridge,
        state: MaterializedForwardState,
        actions: tuple[MemoryAction, ...],
    ) -> None:
        super().__init__()
        self._entrypoint = entrypoint
        self._task = task
        self._function = function
        self._bridge = bridge
        self._state = state
        self._actions = actions

    def forward(self, *arguments: object) -> object:
        leaves, _ = tree_flatten(arguments)
        input_aliases = tuple(
            self._bridge.alias_for_object(slot.object_id)
            for slot in self._entrypoint.input_slots
        )
        stream = torch.cuda.current_stream()
        task_open = False
        try:
            bindings = self._bridge.before_task(
                self._task.task_id, stream, input_aliases
            )
            task_open = True
            for slot, alias_id, binding in zip(
                self._entrypoint.input_slots,
                input_aliases,
                bindings,
                strict=True,
            ):
                tensor = leaves[slot.leaf_index]
                if not isinstance(tensor, torch.Tensor):
                    raise RuntimeError("task tensor input became static")
                self._bridge.rebind(tensor, alias_id, binding)
                self._state.object_store[alias_id] = tensor
                self._state.generations[alias_id] = binding.generation

            # Forward-only execution has no captured backward. Avoid creating
            # hidden dispatcher-autograd contexts across planned task bounds.
            with torch.no_grad():
                output = self._function(*arguments)
            output_leaves, _ = tree_flatten(output)
            produced: set[str] = set()
            for slot in self._entrypoint.output_slots:
                tensor = output_leaves[slot.leaf_index]
                if not isinstance(tensor, torch.Tensor):
                    raise RuntimeError("task tensor output became static")
                alias_id = self._bridge.alias_for_object(slot.object_id)
                if alias_id not in input_aliases and alias_id not in produced:
                    binding = self._bridge.promote_output(alias_id, tensor)
                    self._bridge.rebind(tensor, alias_id, binding)
                    self._state.generations[alias_id] = binding.generation
                    produced.add(alias_id)
                self._state.object_store[alias_id] = tensor
            for action in self._actions:
                if action.kind not in {
                    MemoryActionKind.RELEASE,
                    MemoryActionKind.OFFLOAD,
                }:
                    continue
                alias_id = action.alias_group_id
                tensor = self._state.object_store.get(alias_id)
                generation = self._state.generations.get(alias_id)
                if tensor is None or generation is None:
                    raise RuntimeError(
                        f"action references unbound alias group {alias_id!r}"
                    )
                self._bridge.dematerialize(tensor, alias_id, generation)
            try:
                self._bridge.after_task(
                    self._task.task_id,
                    stream,
                    self._task.mutations,
                    self._actions,
                )
            finally:
                task_open = False
            return output
        except BaseException as error:
            if task_open:
                self._bridge.abort_task_after_failure(
                    f"execute task {self._task.task_id}", error
                )
            raise


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
        trace_labels = {
            entrypoint.task_id: (
                f"execution_{execution_ordinal:06d}.forward."
                f"stage_{execution_ordinal:04d}.{entrypoint.module_target}"
            )
            for execution_ordinal, entrypoint in enumerate(lowered.entrypoints)
        }
        bridge.configure_task_labels(trace_labels)
        for entrypoint in lowered.entrypoints:
            function = functions[entrypoint.artifact.compatibility_digest]
            wrapper = _ExecutingStage(
                entrypoint,
                task_by_id[entrypoint.task_id],
                function,
                bridge,
                state,
                grouped_actions.get(entrypoint.task_id, ()),
            )
            self._root.set_submodule(entrypoint.module_target, wrapper)
        output_objects = {
            item.object_id
            for item in plan.program.objects
            if item.role.value == "output"
        }
        self._caller_output_aliases = tuple(
            dict.fromkeys(
                bridge.alias_for_object(slot.object_id)
                for slot in lowered.entrypoints[-1].output_slots
                if slot.object_id in output_objects
            )
        )
        self._initial_prefetches = tuple(
            item.alias_group_id
            for item in plan.schedule.initial_residency
            if item.location.value == "device"
        )
        self._invocations = 0

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
            initial_actions, task_number=(1 << 60) + self._invocations
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
