"""Exact accumulated-training dispatch through selected AOT graph pairs."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch.utils._pytree import tree_flatten, tree_map

from shadowspill.ir import ExecutionPlan, MemoryAction, MemoryActionKind, TaskSpec

from .capture import GraphArtifact
from .optimizer import OpaqueOptimizerArtifact, current_optimizer_bindings
from .runtime_bridge import RuntimeBridge, actions_by_task
from .training_lowering import LoweredTrainingProgram, TrainingTaskEntrypoint
from .training_materialization import TrainingMaterializedState


@dataclass(frozen=True, slots=True)
class _PlanRun:
    lowered: LoweredTrainingProgram
    plan: ExecutionPlan
    actions: Mapping[str, tuple[MemoryAction, ...]]
    tasks: dict[str, TaskSpec]
    entrypoints: tuple[TrainingTaskEntrypoint, ...]
    initial_device_aliases: tuple[str, ...]
    public_by_microbatch: tuple[tuple[str, ...], ...]
    ephemeral_aliases: frozenset[str]
    objects_by_alias: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class _TensorLayout:
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    dtype: torch.dtype


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

    def __call__(
        self, inputs: Sequence[Sequence[Any]]
    ) -> tuple[tuple[torch.Tensor, ...], tuple[Any, ...]]:
        run = (
            self._initial
            if self._initial is not None and not self._optimizer_state_initialized
            else self._recurrent
        )
        if run is None:
            raise AssertionError("initial optimizer plan is unavailable")
        self._state.refresh_inputs(inputs)
        self._bridge.submit_initial_actions(
            tuple(
                MemoryAction("task_000000", alias_id, MemoryActionKind.PREFETCH)
                for alias_id in run.initial_device_aliases
            ),
            task_number=(1 << 60) + self._invocations,
        )
        public_tensors: dict[int, tuple[torch.Tensor, ...]] = {}
        for entrypoint in run.entrypoints:
            task = run.tasks[entrypoint.task_id]
            if entrypoint.phase == "optimizer":
                self._execute_optimizer(run, entrypoint, task)
                continue
            outputs = self._execute_graph(run, entrypoint, task)
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
        return tuple(losses), tuple(metrics)

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

    def _execute_graph(
        self,
        run: _PlanRun,
        entrypoint: TrainingTaskEntrypoint,
        task: TaskSpec,
    ) -> tuple[torch.Tensor, ...]:
        artifact = entrypoint.artifact
        if not isinstance(artifact, GraphArtifact):
            raise RuntimeError("graph task has no captured artifact")
        function = self._functions[artifact.compatibility_digest]
        stream = torch.cuda.current_stream()
        input_aliases = tuple(
            dict.fromkeys(
                self._bridge.alias_for_object(object_id) for object_id in task.inputs
            )
        )
        task_open = False
        try:
            bindings = self._bridge.before_task(task.task_id, stream, input_aliases)
            task_open = True
            binding_by_alias = dict(zip(input_aliases, bindings, strict=True))
            for alias_id, binding in binding_by_alias.items():
                tensor = self._state.object_store.get(alias_id)
                if tensor is None:
                    raise RuntimeError(f"task input {alias_id!r} has no tensor binding")
                self._bridge.rebind(tensor, alias_id, binding)
                self._state.generations[alias_id] = binding.generation

            arguments = list(artifact.example_arguments)
            for slot in entrypoint.input_slots:
                arguments[slot.leaf_index] = self._state.object_tensors[slot.object_id]
            raw = function(*arguments)
            leaves, _ = tree_flatten(raw)
            if entrypoint.phase == "forward":
                outputs = tuple(
                    value for value in leaves if isinstance(value, torch.Tensor)
                )
                if len(outputs) != len(leaves):
                    raise RuntimeError("captured forward graph returned a static leaf")
                self._bind_forward_outputs(entrypoint, outputs, input_aliases)
            else:
                self._accumulate_gradients(entrypoint, leaves)
                outputs = ()

            self._dematerialize_actions(run, task.task_id)
            self._bridge.after_task(
                task.task_id,
                stream,
                task.mutations,
                run.actions.get(task.task_id, ()),
            )
            task_open = False
            self._forget_released_objects(run, task.task_id)
            return outputs
        except BaseException as error:
            if task_open:
                self._bridge.abort_task_after_failure(
                    f"execute task {task.task_id}", error
                )
            raise

    def _bind_forward_outputs(
        self,
        entrypoint: TrainingTaskEntrypoint,
        outputs: tuple[torch.Tensor, ...],
        input_aliases: tuple[str, ...],
    ) -> None:
        produced: set[str] = set()
        for slot in entrypoint.output_slots:
            tensor = outputs[slot.leaf_index]
            alias_id = self._bridge.alias_for_object(slot.object_id)
            if alias_id not in input_aliases and alias_id not in produced:
                binding = self._bridge.promote_output(alias_id, tensor)
                self._bridge.rebind(tensor, alias_id, binding)
                self._state.generations[alias_id] = binding.generation
                produced.add(alias_id)
            self._state.object_store.setdefault(alias_id, tensor)
            self._state.object_tensors[slot.object_id] = tensor

    def _accumulate_gradients(
        self, entrypoint: TrainingTaskEntrypoint, leaves: list[object]
    ) -> None:
        by_destination: dict[str, tuple[str, list[torch.Tensor]]] = {}
        for slot in entrypoint.gradient_output_slots:
            contribution = leaves[slot.leaf_index]
            if not isinstance(contribution, torch.Tensor):
                raise RuntimeError("parameter gradient became non-tensor")
            alias_id = self._bridge.alias_for_object(slot.object_id)
            item = by_destination.setdefault(alias_id, (slot.object_id, []))
            item[1].append(contribution)

        contributions: list[torch.Tensor] = []
        destinations: list[torch.Tensor] = []
        first: list[tuple[str, str, torch.Tensor]] = []
        for alias_id, (object_id, values) in by_destination.items():
            contribution = values[0]
            for additional in values[1:]:
                contribution.add_(additional)
            destination = self._state.object_store.get(alias_id)
            if destination is None:
                first.append((object_id, alias_id, contribution))
            else:
                destinations.append(destination)
                contributions.append(contribution)
        for object_id, alias_id, contribution in first:
            binding = self._bridge.promote_output(alias_id, contribution)
            self._bridge.rebind(contribution, alias_id, binding)
            self._state.object_store[alias_id] = contribution
            self._state.object_tensors[object_id] = contribution
            self._state.generations[alias_id] = binding.generation
            parameter = self._gradients.get(alias_id)
            if parameter is not None:
                parameter.grad = contribution
        if destinations:
            torch._foreach_add_(destinations, contributions)

    def _execute_optimizer(
        self,
        run: _PlanRun,
        entrypoint: TrainingTaskEntrypoint,
        task: TaskSpec,
    ) -> None:
        stream = torch.cuda.current_stream()
        aliases = tuple(
            dict.fromkeys(
                self._bridge.alias_for_object(object_id) for object_id in task.inputs
            )
        )
        task_open = False
        try:
            bindings = self._bridge.before_task(task.task_id, stream, aliases)
            task_open = True
            for alias_id, binding in zip(aliases, bindings, strict=True):
                tensor = self._state.object_store.get(alias_id)
                if tensor is None:
                    raise RuntimeError(f"optimizer input {alias_id!r} is unbound")
                self._bridge.rebind(tensor, alias_id, binding)
                self._state.generations[alias_id] = binding.generation
            artifact = entrypoint.artifact
            eager = isinstance(artifact, OpaqueOptimizerArtifact) or not (
                self._optimizer_state_available
            )
            if eager:
                with torch.no_grad():
                    self.optimizer.step()
            else:
                if not isinstance(artifact, GraphArtifact):
                    raise RuntimeError("optimizer task has no executable artifact")
                current = self._current_optimizer_bindings()
                try:
                    arguments = tuple(
                        current[name].tensor
                        for name in entrypoint.optimizer_binding_names
                    )
                except KeyError as exc:
                    raise RuntimeError(
                        f"optimizer tensor {exc.args[0]!r} is unbound"
                    ) from exc
                function = self._functions[artifact.compatibility_digest]
                with torch.no_grad():
                    function(*arguments)
            if eager and not self._optimizer_state_available:
                self._bind_created_optimizer_state(run.lowered)
                self._optimizer_state_available = True
            self._dematerialize_actions(run, task.task_id)
            self._bridge.after_task(
                task.task_id,
                stream,
                task.mutations,
                run.actions.get(task.task_id, ()),
            )
            task_open = False
            self._forget_released_objects(run, task.task_id)
            if task.task_id == run.lowered.optimizer_task_id:
                self._optimizer_state_initialized = True
                for parameter in self._gradients.values():
                    parameter.grad = None
                for alias_id in self._gradients:
                    self._state.object_store.pop(alias_id, None)
                    self._state.generations.pop(alias_id, None)
                for gradient_binding in run.lowered.gradients:
                    self._state.object_tensors.pop(
                        gradient_binding.gradient_object_id, None
                    )
        except BaseException as error:
            if task_open:
                self._bridge.abort_task_after_failure(
                    f"execute task {task.task_id}", error
                )
            raise

    def _bind_created_optimizer_state(self, lowered: LoweredTrainingProgram) -> None:
        current = self._current_optimizer_bindings()
        produced: set[str] = set()
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
                binding = self._bridge.promote_output(alias_id, tensor)
                self._bridge.rebind(tensor, alias_id, binding)
                self._state.object_store[alias_id] = tensor
                self._state.generations[alias_id] = binding.generation
                produced.add(alias_id)
            self._state.object_tensors[item.object_id] = tensor

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
                self._bridge.read_host_tensor(alias_id, owner)
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

    def _dematerialize_actions(self, run: _PlanRun, task_id: str) -> None:
        for action in run.actions.get(task_id, ()):
            if action.kind not in {MemoryActionKind.RELEASE, MemoryActionKind.OFFLOAD}:
                continue
            alias_id = action.alias_group_id
            tensor = self._state.object_store.get(alias_id)
            generation = self._state.generations.get(alias_id)
            if tensor is None or generation is None:
                raise RuntimeError(f"action references unbound object {alias_id!r}")
            try:
                self._bridge.dematerialize(tensor, alias_id, generation)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"failed to dematerialize {alias_id!r} after {task_id!r} "
                    f"at generation {generation} from address {tensor.data_ptr()}"
                ) from exc

    def _forget_released_objects(self, run: _PlanRun, task_id: str) -> None:
        for action in run.actions.get(task_id, ()):
            alias_id = action.alias_group_id
            if (
                action.kind is not MemoryActionKind.RELEASE
                or alias_id not in run.ephemeral_aliases
            ):
                continue
            self._state.object_store.pop(alias_id, None)
            self._state.generations.pop(alias_id, None)
            for object_id in run.objects_by_alias.get(alias_id, ()):
                self._state.object_tensors.pop(object_id, None)

    def _prepare(
        self, lowered: LoweredTrainingProgram, plan: ExecutionPlan
    ) -> _PlanRun:
        tasks = {
            item.task_id: item for item in plan.program.selected_tasks(plan.selections)
        }
        entrypoints = tuple(
            item for item in lowered.entrypoints if item.task_id in tasks
        )
        return _PlanRun(
            lowered=lowered,
            plan=plan,
            actions=actions_by_task(plan.schedule.actions),
            tasks=tasks,
            entrypoints=entrypoints,
            initial_device_aliases=tuple(
                item.alias_group_id
                for item in plan.schedule.initial_residency
                if item.location.value == "device"
            ),
            public_by_microbatch=self._public_outputs(entrypoints),
            ephemeral_aliases=frozenset(
                item.alias_group_id
                for item in plan.program.alias_groups
                if item.alias_group_id
                not in {
                    residency.alias_group_id
                    for residency in plan.schedule.initial_residency
                }
            ),
            objects_by_alias={
                alias_id: tuple(
                    item.object_id
                    for item in plan.program.objects
                    if item.alias_group_id == alias_id
                )
                for alias_id in (
                    item.alias_group_id for item in plan.program.alias_groups
                )
            },
        )

    def _public_outputs(
        self, entrypoints: tuple[TrainingTaskEntrypoint, ...]
    ) -> tuple[tuple[str, ...], ...]:
        result: dict[int, tuple[str, ...]] = {}
        for entrypoint in entrypoints:
            if entrypoint.phase != "forward" or entrypoint.microbatch is None:
                continue
            result[entrypoint.microbatch] = tuple(
                self._bridge.alias_for_object(slot.object_id)
                for slot in entrypoint.output_slots[: entrypoint.public_output_count]
            )
        return tuple(result[index] for index in range(len(result)))


__all__ = ["TrainingExecutor"]
