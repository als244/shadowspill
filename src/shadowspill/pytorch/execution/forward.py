"""Ordinary PyTorch task dispatch wrapped by exact runtime boundaries."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils._pytree import TreeSpec, tree_flatten, tree_unflatten

from shadowspill.ir import ExecutionPlan, MemoryAction, MemoryActionKind, TaskSpec
from shadowspill.pytorch.contracts import PlanningError
from shadowspill.pytorch.lowering.forward import LoweredForwardProgram, TaskEntrypoint
from shadowspill.pytorch.materialization.forward import MaterializedForwardState
from shadowspill.pytorch.partition import PartitionedExport
from shadowspill.pytorch.runtime_adapter.bridge import (
    RuntimeBridge,
    TaskMemoryEnvelope,
    actions_by_task,
)
from shadowspill.pytorch.runtime_adapter.failures import ExecutionTaskIdentity
from shadowspill.pytorch.runtime_adapter.fixed_layout import RuntimeFixedLayout
from shadowspill.pytorch.runtime_adapter.transfer_labels import TransferLabelIndex
from shadowspill.pytorch.sharing import ResolvedSharedOutput, TensorRef, format_path


@dataclass(slots=True)
class _PreparedForwardTask:
    arguments: tuple[object, ...]
    input_aliases: tuple[str, ...]
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
        task_handle: int,
    ) -> None:
        super().__init__()
        self._entrypoint = entrypoint
        self._task = task
        self._function = function
        self._bridge = bridge
        self._state = state
        self._actions = actions
        self._identity = identity
        self._task_handle = task_handle
        self._task_index = int(task.task_id.removeprefix("task_"))
        self._device_ordinal = state.device.index or 0
        self._input_aliases = tuple(
            bridge.alias_for_object(slot.object_id)
            for slot in entrypoint.input_slots
        )

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
        runtime_scope_open = False
        try:
            input_tensors: list[torch.Tensor] = []
            for slot in self._entrypoint.input_slots:
                tensor = leaves[slot.leaf_index]
                if not isinstance(tensor, torch.Tensor):
                    raise RuntimeError("task tensor input became static")
                input_tensors.append(tensor)
            generations = self._bridge.before_task_and_acquire(
                self._task_handle,
                self._task_index,
                self._device_ordinal,
                input_tensors,
                self._input_aliases,
            )
            runtime_scope_open = True
            for tensor, alias_id, generation in zip(
                input_tensors,
                self._input_aliases,
                generations,
                strict=True,
            ):
                self._state.object_store[alias_id] = tensor
                self._state.generations[alias_id] = generation
            return _PreparedForwardTask(
                tuple(input_tensors), self._input_aliases
            )
        except BaseException:
            if runtime_scope_open:
                self._bridge.abort_task(self._task_handle, self._task_index)
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
        adopted: list[tuple[torch.Tensor, str]] = []
        replacement_aliases: set[str] = set()
        replacement_leaves = set(self._entrypoint.replacement_output_leaves)
        for slot in self._entrypoint.output_slots:
            tensor = output_leaves[slot.leaf_index]
            if not isinstance(tensor, torch.Tensor):
                raise RuntimeError("task tensor output became static")
            alias_id = self._bridge.alias_for_object(slot.object_id)
            replacement = slot.leaf_index in replacement_leaves
            if replacement and alias_id not in produced:
                adopted.append((tensor, alias_id))
                replacement_aliases.add(alias_id)
                produced.add(alias_id)
            elif alias_id not in prepared.input_aliases and alias_id not in produced:
                adopted.append((tensor, alias_id))
                produced.add(alias_id)
                self._state.object_store[alias_id] = tensor
            elif not replacement:
                self._state.object_store[alias_id] = tensor
        replacements = tuple(
            self._state.replacement_storage_views(alias_id)
            for _tensor, alias_id in adopted
            if alias_id in replacement_aliases
        )
        replacement_by_alias = {item.alias_id: item for item in replacements}
        dematerialized: list[torch.Tensor] = []
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
            if tensor is None or alias_id not in self._state.generations:
                raise RuntimeError(
                    f"action references unbound alias group {alias_id!r}"
                )
            dematerialized.append(tensor)
        generations = self._bridge.after_task_and_update(
            self._task_handle,
            self._task_index,
            self._device_ordinal,
            adopted,
            dematerialized,
            replacements=replacements,
        )
        prepared.runtime_scope_open = False
        for (tensor, alias_id), generation in zip(
            adopted, generations, strict=True
        ):
            if alias_id in replacement_aliases:
                self._state.publish_replacement_generation(
                    replacement_by_alias[alias_id], generation
                )
            else:
                self._state.object_store[alias_id] = tensor
                self._state.generations[alias_id] = generation
        return output

    def _abort_task(
        self,
        prepared: _PreparedForwardTask,
    ) -> None:
        if prepared.runtime_scope_open:
            prepared.runtime_scope_open = False
            self._bridge.abort_task(self._task_handle, self._task_index)


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
        # Materialization uses a short-lived action batch. It is idle now and
        # must not become part of the immutable execution plan.
        bridge.clear_tasks()
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
            task_handle = bridge.admit_task(
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
                task_handle,
            )
            self._root.set_submodule(entrypoint.module_target, wrapper)
        bridge.seal_fixed_layout()
        self._public_output_aliases = tuple(
            bridge.alias_for_object(object_id)
            for object_id in lowered.public_outputs
        )
        shared_indices = {
            item.public_leaf_index for item in self._shared_outputs
        }
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
        self._caller_acquisition_handle = bridge.admit_caller_acquisition(
            self._caller_output_aliases
        )
        self._active_shared_outputs: dict[int, TensorRef] = {}
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
            self._release_closed_shared_output_generations()
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
        public_leaves = [
            output_leaves[index] for index in self._user_output_indices
        ]
        caller_tensors = tuple(
            self._state.object_store[alias_id]
            for alias_id in self._caller_output_aliases
        )
        if self._caller_output_aliases:
            bindings = self._bridge.acquire_for_caller(
                self._caller_output_aliases,
                caller_tensors,
                acquisition_handle=self._caller_acquisition_handle,
            )
            self._bridge.transfer_outputs_to_caller(
                self._caller_output_aliases, caller_tensors, bindings
            )
        created = self._retain_shared_outputs(public_leaves)
        for alias_id in dict.fromkeys(self._public_output_aliases):
            self._state.object_store.pop(alias_id, None)
            self._state.generations.pop(alias_id, None)
        self._invocations += 1
        self._active_shared_outputs = created
        return tree_unflatten(public_leaves, self._output_tree_spec)

    def validate_invocation(self) -> None:
        """Reject a call that would overwrite a still-owned output slot."""

        self._require_shared_output_slots_available()

    def prepare_invocation(
        self, arguments: Sequence[object]
    ) -> Sequence[object]:
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
                object_reference = self._bridge.acquire_object_reference(alias_id)
                try:
                    generation = self._state.generations[alias_id]
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
            self._bridge.release_object_generation(
                alias_id,
                expected_generation=reference.generation,
            )
            released[alias_id] = reference.generation
        self._active_shared_outputs.clear()

    @staticmethod
    def _initial_prefetch_action(alias_id: str) -> MemoryAction:
        return MemoryAction("task_000000", alias_id, MemoryActionKind.PREFETCH)


__all__ = ["ForwardExecutor"]
