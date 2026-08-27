"""Ordinary PyTorch task dispatch wrapped by exact runtime boundaries."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils._pytree import TreeSpec, tree_flatten, tree_unflatten

from shadowspill.errors import PlanningError
from shadowspill.ir import ExecutionPlan, MemoryAction, MemoryActionKind, TaskSpec
from shadowspill.pytorch.invocation import ReusableCompletionEvent
from shadowspill.pytorch.lowering.forward import LoweredForwardProgram, TaskEntrypoint
from shadowspill.pytorch.materialization.forward import MaterializedForwardState
from shadowspill.pytorch.materialization.replacement import ReplacementStorageViews
from shadowspill.pytorch.partition import PartitionedExport
from shadowspill.pytorch.runtime_adapter.bridge import (
    PublishedStorage,
    RuntimeBridge,
    TaskMemoryEnvelope,
    TaskPublication,
    actions_by_task,
)
from shadowspill.pytorch.runtime_adapter.failures import ExecutionTaskIdentity
from shadowspill.pytorch.runtime_adapter.fixed_layout import RuntimeFixedLayout
from shadowspill.pytorch.runtime_adapter.transfer_labels import TransferLabelIndex
from shadowspill.pytorch.sharing import ResolvedSharedOutput, TensorRef, format_path

from .annotations import AnnotatedExecutor, TaskBoundaryAnnotations


@dataclass(slots=True)
class _PreparedForwardTask:
    arguments: tuple[object, ...]
    input_aliases: tuple[str, ...]
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
        alias_id = bridge.alias_for_object(slot.object_id)
        replace_lease = slot.leaf_index in replacement_leaves
        adopt = (replace_lease or alias_id not in input_aliases) and (
            alias_id not in produced
        )
        if not adopt:
            continue
        produced.add(alias_id)
        if bridge.requires_storage(alias_id):
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
        self._annotations = annotations
        self._trace_label = f"{identity.execution_task_id}.{identity.semantic_name}"
        self._publication_ordinals = {
            item.alias_id: ordinal for ordinal, item in enumerate(publications)
        }
        self._device_ordinal = state.device.index or 0
        self._input_aliases = tuple(
            bridge.alias_for_object(slot.object_id) for slot in entrypoint.input_slots
        )
        self._input_storage_indices = tuple(
            index
            for index, alias_id in enumerate(self._input_aliases)
            if bridge.requires_storage(alias_id)
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
        runtime_scope_open = False
        try:
            with self._annotations.range(
                f"shadowspill.before_task.{self._trace_label}"
            ):
                input_tensors = self._resolve_inputs(arguments)
                self._bridge.before_task_and_acquire(
                    self._task_handle,
                    self._device_ordinal,
                    tuple(
                        input_tensors[index] for index in self._input_storage_indices
                    ),
                )
                runtime_scope_open = True
                self._publish_input_bindings(input_tensors)
                prepared = _PreparedForwardTask(input_tensors, self._input_aliases)
            return prepared
        except BaseException:
            if runtime_scope_open:
                self._bridge.abort_task(self._task_handle)
            raise

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

    def _run_compiled_task(self, prepared: _PreparedForwardTask) -> object:
        # Forward-only execution has no captured backward. Avoid creating
        # hidden dispatcher-autograd problems across planned task bounds.
        with (
            self._annotations.range(f"shadowspill.compiled_call.{self._trace_label}"),
            torch.no_grad(),
        ):
            return self._function(*prepared.arguments)

    def _after_task(
        self,
        prepared: _PreparedForwardTask,
        output: object,
    ) -> object:
        with self._annotations.range(f"shadowspill.after_task.{self._trace_label}"):
            processed = self._process_outputs(output)
            self._bridge.after_task_and_update(
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
            alias_id = self._bridge.alias_for_object(slot.object_id)
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
            self._bridge.abort_task(self._task_handle)


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
        transfer_labels = TransferLabelIndex(plan.program, trace_labels)
        for execution_ordinal, entrypoint in enumerate(lowered.entrypoints):
            task = task_by_id[entrypoint.task_id]
            task_actions = grouped_actions.get(entrypoint.task_id, ())
            input_aliases = tuple(
                bridge.alias_for_object(slot.object_id)
                for slot in entrypoint.input_slots
            )
            publications = _forward_publications(entrypoint, input_aliases, bridge)
            task_handle = bridge.admit_task(
                task,
                input_aliases,
                task_actions,
                transfer_labels.labels_for(task_actions),
                memory_envelopes.get(task.task_id, TaskMemoryEnvelope()),
                trace_label=trace_labels[entrypoint.task_id],
                publications=publications,
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
                publications,
                self._task_annotations,
            )
            self._root.set_submodule(entrypoint.module_target, wrapper)
        bridge.seal_fixed_layout()
        self._public_output_aliases = tuple(
            bridge.alias_for_object(object_id) for object_id in lowered.public_outputs
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
        self._caller_acquisition_handle = bridge.admit_caller_acquisition(
            self._caller_output_aliases
        )
        self._active_shared_outputs: dict[int, TensorRef] = {}
        self._completion = ReusableCompletionEvent(state.device)
        self._initial_task_id = fixed_layout.initial_task_id
        self._invocations = 0

    def __call__(self, arguments: Sequence[object]) -> object:
        if self._invocations:
            # Forward v1 is also non-cyclic: begin only after the preceding
            # invocation reaches its declared terminal residency.
            self._bridge.wait_plan_idle()
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
        public_leaves = [output_leaves[index] for index in self._user_output_indices]
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
                object_reference = self._bridge.acquire_object_reference(alias_id)
                try:
                    generation = self._bridge.current_generation(alias_id)
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
