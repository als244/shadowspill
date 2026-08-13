"""Incremental storage ownership for planned accumulated training."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.export.graph_signature import InputKind

from shadowspill.ir import MemoryAction, MemoryActionKind

from ._live_storage import unique_live_tensors
from .aot import TrainingObjectiveCapture
from .contracts import PlanningError
from .lowering import RegistrationBinding
from .optimizer import current_optimizer_bindings
from .runtime_bridge import RuntimeBridge
from .training_lowering import LoweredTrainingProgram, TrainingStorageLayout


@dataclass(frozen=True, slots=True)
class _Registration:
    binding: RegistrationBinding
    tensor: torch.Tensor


class TrainingMaterializedState:
    """Own model, input, and fixed storages while preserving Tensor identities."""

    def __init__(
        self,
        model: nn.Module,
        layout: TrainingStorageLayout,
        captures: tuple[TrainingObjectiveCapture, ...],
        example_inputs: tuple[Sequence[Any], ...],
        bridge: RuntimeBridge,
        *,
        device_ordinal: int,
    ) -> None:
        self.model = model
        self.layout = layout
        self.captures = captures
        self.bridge = bridge
        self.device = torch.device("cuda", device_ordinal)
        self.object_store: dict[str, torch.Tensor] = {}
        self.object_tensors: dict[str, torch.Tensor] = {}
        self.generations: dict[str, int] = {}
        self._closed = False
        self._model_on_cpu = False
        self._state_names = tuple(model.state_dict().keys())
        self._model_aliases: set[str] = set()
        self._input_aliases: set[str] = set()
        self._user_alias_by_position: tuple[dict[int, str], ...] = ()
        self._planning_cpu_owners: dict[str, torch.Tensor] = {}
        self._object_ids_by_alias: dict[str, tuple[str, ...]] = {
            group.alias_group_id: tuple(
                item.object_id
                for item in layout.program.objects
                if item.alias_group_id == group.alias_group_id
            )
            for group in layout.program.alias_groups
        }
        self._flat_arguments = tuple(
            _flat_training_arguments(capture, model, microbatch)
            for capture, microbatch in zip(captures, example_inputs, strict=True)
        )
        try:
            self._materialize_initial()
        except BaseException:
            self.restore_cpu_and_unregister()
            raise

    def restore_model_cpu_for_optimizer_capture(self) -> None:
        """Temporarily expose preserved CPU bytes without replacing Parameters."""

        if self._closed or self._model_on_cpu:
            return
        for item in self._registrations():
            alias_id = self.bridge.alias_for_object(item.binding.object_id)
            item.tensor.data = self._cpu_view(
                self._planning_cpu_owners[alias_id], item.tensor
            )
        self._model_on_cpu = True

    def restore_cuda_placeholders_after_optimizer_capture(self) -> None:
        """Return registered model tensors to host-only CUDA placeholders."""

        if self._closed or not self._model_on_cpu:
            return
        registrations: dict[str, list[_Registration]] = {}
        for item in self._registrations():
            alias_id = self.bridge.alias_for_object(item.binding.object_id)
            registrations.setdefault(alias_id, []).append(item)
        for ordinal, (alias_id, items) in enumerate(registrations.items()):
            owner = torch.empty(
                self._planning_cpu_owners[alias_id].numel(),
                dtype=torch.uint8,
                device=self.device,
            )
            assigned: set[int] = set()
            representative: torch.Tensor = owner
            for item in items:
                if id(item.tensor) in assigned:
                    continue
                view = self._cuda_view(owner, item.tensor)
                item.tensor.data = view
                representative = item.tensor
                assigned.add(id(item.tensor))
            binding = self.bridge.bind_registered_tensor(alias_id, owner)
            self.object_store[alias_id] = representative
            for item in items:
                self.object_tensors[item.binding.object_id] = item.tensor
            self.generations[alias_id] = binding.generation
            self._release_placeholder(
                alias_id, representative, binding.generation, ordinal
            )
        self.bridge.wait_idle()
        self._model_on_cpu = False

    def adopt_execution_plan(
        self,
        bridge: RuntimeBridge,
        lowered: LoweredTrainingProgram,
        *,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Switch from provisional identities and install fixed graph inputs."""

        existing = self.bridge.registered_aliases()
        bridge.adopt_registered(existing)
        self.bridge = bridge
        active_objects = {
            object_id
            for task in lowered.program.tasks
            for object_id in (*task.inputs, *task.outputs)
        }
        for ordinal, fixed in enumerate(lowered.fixed_tensors):
            if fixed.object_id not in active_objects:
                continue
            alias_id = bridge.alias_for_object(fixed.object_id)
            if alias_id in bridge.registered_aliases():
                continue
            source = torch.empty_strided(
                tuple(fixed.value.shape),
                tuple(fixed.value.stride()),
                dtype=fixed.value.dtype,
                device="cpu",
            )
            source.fill_(1)
            bridge.register_host_tensor(alias_id, source, retain_spill_copy=True)
            owner = torch.empty(
                source.untyped_storage().nbytes(),
                dtype=torch.uint8,
                device=self.device,
            )
            tensor = self._cuda_view(owner, source)
            binding = bridge.bind_registered_tensor(alias_id, owner)
            self.object_store[alias_id] = tensor
            self.object_tensors[fixed.object_id] = tensor
            self.generations[alias_id] = binding.generation
            self._release_placeholder(
                alias_id,
                tensor,
                binding.generation,
                (1 << 20) + ordinal,
            )
        self._materialize_optimizer_state(optimizer, lowered)
        bridge.wait_idle()

    def _materialize_optimizer_state(
        self,
        optimizer: torch.optim.Optimizer,
        lowered: LoweredTrainingProgram,
    ) -> None:
        current = {
            item.name: item
            for item in current_optimizer_bindings(
                dict(self.model.named_parameters()), optimizer
            )
        }
        entries: dict[str, list[tuple[str, torch.Tensor]]] = {}
        for item in lowered.optimizer_objects:
            actual = current.get(item.name)
            if actual is None:
                if item.created_on_first_step:
                    continue
                raise PlanningError(
                    f"optimizer state {item.name!r} is absent after initialization"
                )
            alias_id = self.bridge.alias_for_object(item.object_id)
            entries.setdefault(alias_id, []).append((item.object_id, actual.tensor))

        for ordinal, (alias_id, values) in enumerate(entries.items()):
            source = values[0][1]
            if source.device.type != "cpu":
                raise PlanningError(
                    f"optimizer state for {alias_id!r} must begin on the CPU"
                )
            source_owner = torch.empty(0, dtype=torch.uint8, device="cpu")
            source_owner.set_(source.untyped_storage())
            self.bridge.register_host_tensor(
                alias_id, source_owner, retain_spill_copy=True
            )
            owner = torch.empty(
                source_owner.numel(), dtype=torch.uint8, device=self.device
            )
            representative: torch.Tensor = owner
            assigned: set[int] = set()
            for object_id, tensor in values:
                if id(tensor) not in assigned:
                    tensor.data = self._cuda_view(owner, tensor)
                    assigned.add(id(tensor))
                self.object_tensors[object_id] = tensor
                representative = tensor
            binding = self.bridge.bind_registered_tensor(alias_id, owner)
            self.object_store[alias_id] = representative
            self.generations[alias_id] = binding.generation
            self._release_placeholder(
                alias_id,
                representative,
                binding.generation,
                (1 << 21) + ordinal,
            )

    def refresh_inputs(self, values: Sequence[Sequence[Any]]) -> None:
        """Write every guarded microbatch into its persistent host slot."""

        self.bridge.wait_idle()
        for capture, microbatch, slots in zip(
            self.captures,
            values,
            self._user_alias_by_position,
            strict=True,
        ):
            flat = _flat_training_arguments(capture, self.model, microbatch)
            written: set[str] = set()
            for index, alias_id in slots.items():
                tensor = flat[index]
                if not isinstance(tensor, torch.Tensor):
                    raise RuntimeError("captured tensor input became static")
                if tensor.device.type == "cuda":
                    tensor = tensor.detach().cpu()
                if alias_id not in written:
                    self.bridge.write_spill_tensor(alias_id, tensor)
                    written.add(alias_id)

    def replace_alias_generation(
        self,
        alias_id: str,
        target: torch.Tensor,
        target_generation: int,
    ) -> None:
        """Rebind every persistent frontend view to a functional replacement."""

        previous_generation = self.generations.get(alias_id)
        if previous_generation is None:
            raise RuntimeError(f"replacement alias {alias_id!r} has no generation")
        tensors: list[torch.Tensor] = []
        representative = self.object_store.get(alias_id)
        if representative is not None:
            tensors.append(representative)
        for object_id in self._object_ids_by_alias.get(alias_id, ()):
            tensor = self.object_tensors.get(object_id)
            if tensor is not None:
                tensors.append(tensor)
        unique = unique_live_tensors(tensors)
        if not unique:
            raise RuntimeError(f"replacement alias {alias_id!r} has no frontend view")
        self.bridge.replace_many(
            unique,
            alias_id,
            previous_generation=previous_generation,
            target_tensor=target,
            target_generation=target_generation,
        )
        self.object_store[alias_id] = unique[0]
        self.generations[alias_id] = target_generation

    def state_dict(self) -> OrderedDict[str, torch.Tensor]:
        if self._closed:
            return OrderedDict(self.model.state_dict())
        owners = self._read_model_aliases()
        result: OrderedDict[str, torch.Tensor] = OrderedDict()
        for item in self._registrations():
            if item.binding.name not in self._state_names:
                continue
            alias_id = self.bridge.alias_for_object(item.binding.object_id)
            result[item.binding.name] = self._cpu_view(owners[alias_id], item.tensor)
        missing = set(self._state_names) - set(result)
        if missing:
            raise RuntimeError(f"unsupported model state entries: {sorted(missing)}")
        return result

    def load_model_state(self, state: Mapping[str, torch.Tensor]) -> None:
        expected = set(self._state_names)
        if set(state) != expected:
            raise RuntimeError("model state_dict keys differ")
        owners = self._read_model_aliases()
        for item in self._registrations():
            if item.binding.name not in expected:
                continue
            source = state[item.binding.name]
            if not isinstance(source, torch.Tensor):
                raise TypeError(
                    f"model state entry {item.binding.name!r} must be a tensor"
                )
            destination = self._cpu_view(
                owners[self.bridge.alias_for_object(item.binding.object_id)],
                item.tensor,
            )
            if source.shape != destination.shape or source.dtype != destination.dtype:
                raise RuntimeError(
                    f"model state entry {item.binding.name!r} has incompatible geometry"
                )
            destination.copy_(source.detach().to(device="cpu"))
        for alias_id, owner in owners.items():
            self.bridge.write_spill_tensor(alias_id, owner)

    def restore_cpu_and_unregister(self) -> None:
        if self._closed:
            return
        self.bridge.wait_idle()
        owners = (
            self._planning_cpu_owners
            if self._model_on_cpu
            else self._read_model_aliases()
        )
        assigned: set[int] = set()
        for item in self._registrations():
            if id(item.tensor) in assigned:
                continue
            alias_id = self.bridge.alias_for_object(item.binding.object_id)
            item.tensor.data = self._cpu_view(owners[alias_id], item.tensor)
            assigned.add(id(item.tensor))
        self.bridge.unregister(self.bridge.registered_aliases())
        self.object_store.clear()
        self.object_tensors.clear()
        self.generations.clear()
        self._closed = True
        self._model_on_cpu = True

    def _materialize_initial(self) -> None:
        registrations = self._registrations()
        registration_by_id = {id(item.tensor): item for item in registrations}
        entries: dict[str, list[tuple[torch.Tensor, tuple[int, int] | None]]] = {}
        for item in registrations:
            alias_id = self.bridge.alias_for_object(item.binding.object_id)
            self._model_aliases.add(alias_id)
            entries.setdefault(alias_id, []).append((item.tensor, None))
        slot_maps: list[dict[int, str]] = []
        for position, (flat, root_slots) in enumerate(
            zip(self._flat_arguments, self.layout.root_input_slots, strict=True)
        ):
            slots: dict[int, str] = {}
            for slot in root_slots:
                value = flat[slot.leaf_index]
                if not isinstance(value, torch.Tensor):
                    raise PlanningError("training tensor slot became static")
                alias_id = self.bridge.alias_for_object(slot.object_id)
                if alias_id not in self._model_aliases:
                    self._input_aliases.add(alias_id)
                    slots[slot.leaf_index] = alias_id
                    entries.setdefault(alias_id, []).append(
                        (value, (position, slot.leaf_index))
                    )
            slot_maps.append(slots)
        self._user_alias_by_position = tuple(slot_maps)

        retain = {
            group.alias_group_id: group.retain_spill_copy
            for group in self.layout.program.alias_groups
        }
        for ordinal, alias_id in enumerate(
            group.alias_group_id for group in self.layout.program.alias_groups
        ):
            values = entries.get(alias_id)
            if not values:
                continue
            source = values[0][0]
            self.bridge.register_host_tensor(
                alias_id, source, retain_spill_copy=retain[alias_id]
            )
            cpu_owner = torch.empty(0, dtype=torch.uint8, device="cpu")
            cpu_owner.set_(source.untyped_storage())
            if alias_id in self._model_aliases:
                self._planning_cpu_owners[alias_id] = cpu_owner
            owner = torch.empty(
                source.untyped_storage().nbytes(), dtype=torch.uint8, device=self.device
            )
            views: dict[int, torch.Tensor] = {}
            representative: torch.Tensor = owner
            for source_tensor, root_position in values:
                view = views.get(id(source_tensor))
                if view is None:
                    view = self._cuda_view(owner, source_tensor)
                    registered = registration_by_id.get(id(source_tensor))
                    if registered is not None:
                        source_tensor.data = view
                        view = source_tensor
                    views[id(source_tensor)] = view
                if root_position is None:
                    registered = registration_by_id.get(id(source_tensor))
                    if registered is not None:
                        self.object_tensors[registered.binding.object_id] = view
                representative = view
                if root_position is not None:
                    position, index = root_position
                    slot = next(
                        item
                        for item in self.layout.root_input_slots[position]
                        if item.leaf_index == index
                    )
                    self.object_tensors[slot.object_id] = view
                    mutable = list(self._flat_arguments[position])
                    mutable[index] = view
                    self._flat_arguments = (
                        *self._flat_arguments[:position],
                        tuple(mutable),
                        *self._flat_arguments[position + 1 :],
                    )
            binding = self.bridge.bind_registered_tensor(alias_id, owner)
            self.object_store[alias_id] = representative
            self.generations[alias_id] = binding.generation
            self._release_placeholder(
                alias_id, representative, binding.generation, ordinal
            )
        self.bridge.wait_idle()

    def _release_placeholder(
        self, alias_id: str, tensor: torch.Tensor, generation: int, ordinal: int
    ) -> None:
        self.bridge.dematerialize(tensor, alias_id, generation)
        self.bridge.submit_initial_actions(
            (MemoryAction("task_000000", alias_id, MemoryActionKind.RELEASE),),
            task_number=(1 << 61) + ordinal,
        )

    def _read_model_aliases(self) -> dict[str, torch.Tensor]:
        self.bridge.wait_idle()
        owners = self._empty_model_aliases()
        for alias_id, owner in owners.items():
            self.bridge.read_spill_tensor(alias_id, owner)
        return owners

    def _empty_model_aliases(self) -> dict[str, torch.Tensor]:
        sizes = {
            item.alias_group_id: item.size_bytes
            for item in self.layout.program.alias_groups
        }
        return {
            alias_id: torch.empty(sizes[alias_id], dtype=torch.uint8, device="cpu")
            for alias_id in self._model_aliases
        }

    def _registrations(self) -> tuple[_Registration, ...]:
        return tuple(
            _Registration(
                binding,
                self.model.get_parameter(binding.name)
                if binding.parameter
                else self.model.get_buffer(binding.name),
            )
            for binding in self.layout.registrations
        )

    @staticmethod
    def _cuda_view(owner: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        view = torch.empty(0, dtype=source.dtype, device=owner.device)
        view.set_(
            owner.untyped_storage(),
            source.storage_offset(),
            tuple(source.shape),
            tuple(source.stride()),
        )
        view.requires_grad_(source.requires_grad)
        return view

    @staticmethod
    def _cpu_view(owner: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
        return torch.empty(0, dtype=tensor.dtype, device="cpu").set_(
            owner.untyped_storage(),
            tensor.storage_offset(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
        )


def _flat_training_arguments(
    capture: TrainingObjectiveCapture,
    model: nn.Module,
    microbatch: Sequence[Any],
) -> tuple[object, ...]:
    exported = capture.exported.exported_program
    flatten = getattr(exported, "_graph_module_flat_inputs", None)
    if not callable(flatten):
        raise PlanningError("PyTorch Export flat-input adapter is unavailable")
    flat = list(flatten(tuple(microbatch), {}))
    for index, spec in enumerate(exported.graph_signature.input_specs):
        if spec.kind not in {InputKind.PARAMETER, InputKind.BUFFER}:
            continue
        if spec.target is None or not spec.target.startswith("model."):
            raise PlanningError("training state target is not rooted at model")
        target = spec.target.removeprefix("model.")
        flat[index] = (
            model.get_parameter(target)
            if spec.kind is InputKind.PARAMETER
            else model.get_buffer(target)
        )
    return tuple(flat)


def representative_training_arguments(
    capture: TrainingObjectiveCapture,
    model: nn.Module,
    microbatch: Sequence[Any],
) -> tuple[object, ...]:
    """Expose authentic root values for isolated task profiling."""

    return tuple(
        value.detach() if isinstance(value, torch.Tensor) else value
        for value in _flat_training_arguments(capture, model, microbatch)
    )


__all__ = ["TrainingMaterializedState", "representative_training_arguments"]
