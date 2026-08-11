"""Exact accumulated-training dispatch through selected AOT graph pairs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch.utils._pytree import tree_flatten

from shadowspill.ir import ExecutionPlan, MemoryAction, MemoryActionKind, TaskSpec

from .runtime_bridge import RuntimeBridge, actions_by_task
from .training_lowering import LoweredTrainingProgram, TrainingTaskEntrypoint
from .training_materialization import TrainingMaterializedState


class TrainingExecutor:
    """Execute selected forward/backward variants and one optimizer update."""

    def __init__(
        self,
        lowered: LoweredTrainingProgram,
        plan: ExecutionPlan,
        bridge: RuntimeBridge,
        state: TrainingMaterializedState,
        functions: dict[str, Callable[..., object]],
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self._lowered = lowered
        self._plan = plan
        self._bridge = bridge
        self._state = state
        self._functions = functions
        self.optimizer = optimizer
        self._actions = actions_by_task(plan.schedule.actions)
        self._tasks = {
            item.task_id: item for item in plan.program.selected_tasks(plan.selections)
        }
        self._entrypoints = tuple(
            item for item in lowered.entrypoints if item.task_id in self._tasks
        )
        self._gradients = {
            state.bridge.alias_for_object(item.gradient_object_id): model_parameter
            for item in lowered.gradients
            for model_parameter in (state.model.get_parameter(item.parameter_name),)
        }
        self._initial_device_aliases = tuple(
            item.alias_group_id
            for item in plan.schedule.initial_residency
            if item.location.value == "device"
        )
        self._public_by_microbatch = self._public_outputs()
        self._invocations = 0

    def __call__(
        self, inputs: Sequence[Sequence[Any]]
    ) -> tuple[tuple[torch.Tensor, ...], tuple[Any, ...]]:
        self._state.refresh_inputs(inputs)
        self._bridge.submit_initial_actions(
            tuple(
                MemoryAction("task_000000", alias_id, MemoryActionKind.PREFETCH)
                for alias_id in self._initial_device_aliases
            ),
            task_number=(1 << 60) + self._invocations,
        )
        public_tensors: dict[int, tuple[torch.Tensor, ...]] = {}
        for entrypoint in self._entrypoints:
            task = self._tasks[entrypoint.task_id]
            if entrypoint.phase == "optimizer":
                self._execute_optimizer(task)
                continue
            outputs = self._execute_graph(entrypoint, task)
            if entrypoint.phase == "forward" and entrypoint.microbatch is not None:
                public_tensors[entrypoint.microbatch] = outputs[
                    : entrypoint.public_output_count
                ]
        ordered = tuple(public_tensors[index] for index in range(len(public_tensors)))
        aliases = tuple(
            alias_id for values in self._public_by_microbatch for alias_id in values
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

    def _execute_graph(
        self, entrypoint: TrainingTaskEntrypoint, task: TaskSpec
    ) -> tuple[torch.Tensor, ...]:
        artifact = entrypoint.artifact
        if artifact is None:
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

            self._dematerialize_actions(task.task_id)
            self._bridge.after_task(
                task.task_id,
                stream,
                task.mutations,
                self._actions.get(task.task_id, ()),
            )
            task_open = False
            return outputs
        except BaseException:
            if task_open:
                self._bridge.abort_task()
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
        contributions: list[torch.Tensor] = []
        destinations: list[torch.Tensor] = []
        first: list[tuple[str, str, torch.Tensor]] = []
        for slot in entrypoint.gradient_output_slots:
            contribution = leaves[slot.leaf_index]
            if not isinstance(contribution, torch.Tensor):
                raise RuntimeError("parameter gradient became non-tensor")
            alias_id = self._bridge.alias_for_object(slot.object_id)
            destination = self._state.object_store.get(alias_id)
            if destination is None:
                first.append((slot.object_id, alias_id, contribution))
            else:
                destinations.append(destination)
                contributions.append(contribution)
        for object_id, alias_id, contribution in first:
            binding = self._bridge.promote_output(alias_id, contribution)
            self._bridge.rebind(contribution, alias_id, binding)
            self._state.object_store[alias_id] = contribution
            self._state.object_tensors[object_id] = contribution
            self._state.generations[alias_id] = binding.generation
            self._gradients[alias_id].grad = contribution
        if destinations:
            torch._foreach_add_(destinations, contributions)

    def _execute_optimizer(self, task: TaskSpec) -> None:
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
            with torch.no_grad():
                self.optimizer.step()
            self._dematerialize_actions(task.task_id)
            self._bridge.after_task(
                task.task_id,
                stream,
                task.mutations,
                self._actions.get(task.task_id, ()),
            )
            task_open = False
            for parameter in self._gradients.values():
                parameter.grad = None
            for alias_id in self._gradients:
                self._state.object_store.pop(alias_id, None)
                self._state.generations.pop(alias_id, None)
            for gradient_binding in self._lowered.gradients:
                self._state.object_tensors.pop(
                    gradient_binding.gradient_object_id, None
                )
        except BaseException:
            if task_open:
                self._bridge.abort_task()
            raise

    def _dematerialize_actions(self, task_id: str) -> None:
        for action in self._actions.get(task_id, ()):
            if action.kind not in {MemoryActionKind.RELEASE, MemoryActionKind.OFFLOAD}:
                continue
            alias_id = action.alias_group_id
            tensor = self._state.object_store.get(alias_id)
            generation = self._state.generations.get(alias_id)
            if tensor is None or generation is None:
                raise RuntimeError(f"action references unbound object {alias_id!r}")
            self._bridge.dematerialize(tensor, alias_id, generation)

    def _public_outputs(self) -> tuple[tuple[str, ...], ...]:
        result: dict[int, tuple[str, ...]] = {}
        for entrypoint in self._entrypoints:
            if entrypoint.phase != "forward" or entrypoint.microbatch is None:
                continue
            result[entrypoint.microbatch] = tuple(
                self._bridge.alias_for_object(slot.object_id)
                for slot in entrypoint.output_slots[: entrypoint.public_output_count]
            )
        return tuple(result[index] for index in range(len(result)))


__all__ = ["TrainingExecutor"]
