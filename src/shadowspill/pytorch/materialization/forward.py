"""Incremental CPU-state materialization through allocator-owned CUDA storage."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.export.graph_signature import InputKind
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

from shadowspill.errors import InputGuardError, PlanningError
from shadowspill.ir import MemoryAction, MemoryActionKind
from shadowspill.pytorch.capture.aot import ExportCapture
from shadowspill.pytorch.capture.live_storage import unique_live_tensors
from shadowspill.pytorch.contracts import TensorSpec
from shadowspill.pytorch.lowering.catalog import RegistrationBinding
from shadowspill.pytorch.lowering.forward import LoweredForwardProgram
from shadowspill.pytorch.materialization.replacement import ReplacementStorageViews
from shadowspill.pytorch.runtime_adapter.bridge import RuntimeBridge
from shadowspill.pytorch.runtime_adapter.runtime import Runtime
from shadowspill.pytorch.sharing import (
    ResolvedSharedInput,
    SharedInput,
    TensorRef,
)
from shadowspill.pytorch.state.storage import (
    adopt_persistent_tensor,
    persistent_state,
    restore_persistent_state,
)


def representative_cpu_inputs(values: Any) -> Any:
    """Materialize TensorSpec/meta leaves without changing authentic CPU values."""

    synthetic_ordinal = 0

    def convert(value: object) -> object:
        nonlocal synthetic_ordinal
        if isinstance(value, TensorSpec):
            result = torch.empty_strided(
                value.shape, value.resolved_stride, dtype=value.dtype, device="cpu"
            )
            result.requires_grad_(value.requires_grad)
        if isinstance(value, torch.Tensor) and value.device.type == "meta":
            result = torch.empty_strided(
                tuple(value.shape),
                tuple(value.stride()),
                dtype=value.dtype,
                device="cpu",
            )
            result.requires_grad_(value.requires_grad)
        elif not isinstance(value, TensorSpec):
            return value
        if not result.is_floating_point() and not result.is_complex():
            raise PlanningError(
                "integer/boolean planning inputs require authentic "
                "caller-supplied tensor values; TensorSpec and meta tensors "
                "provide geometry only"
            )
        generator = torch.Generator(device="cpu").manual_seed(
            0x5A17_0000 + synthetic_ordinal
        )
        synthetic_ordinal += 1
        with torch.no_grad():
            if result.is_floating_point():
                result.copy_(
                    torch.randn(
                        tuple(result.shape),
                        dtype=torch.float32,
                        generator=generator,
                    ).to(dtype=result.dtype)
                )
            elif result.is_complex():
                real = torch.randn(
                    tuple(result.shape), dtype=torch.float32, generator=generator
                )
                imaginary = torch.randn(
                    tuple(result.shape), dtype=torch.float32, generator=generator
                )
                result.copy_(torch.complex(real, imaginary).to(dtype=result.dtype))
        return result

    return tree_map(convert, values)


def flat_runtime_arguments(
    capture: ExportCapture,
    model: nn.Module,
    inputs: Sequence[Any],
) -> list[object]:
    """Combine original registered state with one guarded user invocation."""

    flatten = getattr(capture.exported_program, "_graph_module_flat_inputs", None)
    if not callable(flatten):
        raise PlanningError("PyTorch Export flat-input adapter is unavailable")
    flat = list(flatten(tuple(inputs), {}))
    specs = capture.exported_program.graph_signature.input_specs
    if len(flat) != len(specs):
        raise PlanningError("Export flat input count changed after capture")
    for index, spec in enumerate(specs):
        if spec.kind is InputKind.PARAMETER:
            if spec.target is None:
                raise PlanningError("Export parameter input has no target")
            flat[index] = model.get_parameter(spec.target)
        elif spec.kind is InputKind.BUFFER:
            if spec.target is None:
                raise PlanningError("Export buffer input has no target")
            flat[index] = model.get_buffer(spec.target)
        elif spec.kind is InputKind.CONSTANT_TENSOR:
            if spec.target is None:
                raise PlanningError("Export constant tensor has no target")
            value = _resolve_attribute(model, spec.target)
            if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
                raise PlanningError(
                    f"constant tensor {spec.target!r} is not CPU resident"
                )
            flat[index] = value
    return flat


def _resolve_attribute(module: nn.Module, target: str) -> object:
    value: object = module
    for component in target.split("."):
        if not hasattr(value, component):
            raise PlanningError(f"model has no exported attribute {target!r}")
        value = getattr(value, component)
    return value


@dataclass(frozen=True, slots=True)
class _Registration:
    binding: RegistrationBinding
    tensor: torch.Tensor


type _MaterializationEntry = tuple[torch.Tensor, int | None]


class MaterializedForwardState:
    """Own CUDA placeholders and restore original registered objects on close."""

    def __init__(
        self,
        model: nn.Module,
        lowered: LoweredForwardProgram,
        capture: ExportCapture,
        example_inputs: Sequence[Any],
        bridge: RuntimeBridge,
        *,
        runtime: Runtime,
        device_ordinal: int,
        shared_inputs: tuple[ResolvedSharedInput, ...] = (),
    ) -> None:
        self.model = model
        self.lowered = lowered
        self.capture = capture
        self.bridge = bridge
        self.runtime = runtime
        self._shared_inputs = tuple(shared_inputs)
        self.device = torch.device("cuda", device_ordinal)
        self.root_arguments = flat_runtime_arguments(capture, model, example_inputs)
        self.object_store: dict[str, torch.Tensor] = {}
        self._closed = False
        self._state_names = tuple(model.state_dict().keys())
        self._model_aliases: set[str] = set()
        self._registered_model_aliases: set[str] = set()
        self._persistent_aliases: set[str] = set()
        self._persistent_state = persistent_state(runtime, model)
        self._user_alias_by_position: dict[int, str] = {}
        self._shared_alias_by_position: dict[int, str] = {}
        self._object_ids_by_alias: dict[str, tuple[str, ...]] = {
            group.alias_group_id: tuple(
                item.object_id
                for item in lowered.program.objects
                if item.alias_group_id == group.alias_group_id
            )
            for group in lowered.program.alias_groups
        }
        try:
            self._materialize()
        except BaseException:
            self.restore_cpu_and_unregister()
            raise

    def state_dict(self) -> OrderedDict[str, torch.Tensor]:
        """Return a synchronous CPU snapshot with ordinary state-dict names."""

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
            raise RuntimeError(
                "model state contains unsupported non-parameter entries: "
                f"{sorted(missing)}"
            )
        return result

    def load_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        """Replace host-authoritative model bytes after exact key validation."""

        if self._closed:
            self.model.load_state_dict(state)
            return
        expected = set(self._state_names)
        supplied = set(state)
        if expected != supplied:
            missing = sorted(expected - supplied)
            unexpected = sorted(supplied - expected)
            raise RuntimeError(
                f"state_dict keys differ: missing={missing}, unexpected={unexpected}"
            )
        owners = self._read_model_aliases()
        for item in self._registrations():
            name = item.binding.name
            if name not in expected:
                continue
            source = state[name]
            if not isinstance(source, torch.Tensor):
                raise TypeError(f"state_dict entry {name!r} must be a tensor")
            destination = self._cpu_view(
                owners[self.bridge.alias_for_object(item.binding.object_id)],
                item.tensor,
            )
            if (
                tuple(source.shape) != tuple(destination.shape)
                or source.dtype != destination.dtype
            ):
                raise RuntimeError(
                    f"state_dict entry {name!r} has incompatible shape or dtype"
                )
            destination.copy_(source.detach().to(device="cpu"))
        for alias_id, owner in owners.items():
            self.bridge.write_spill_tensor(alias_id, owner)

    def refresh_inputs(self, inputs: Sequence[Any]) -> tuple[object, ...]:
        """Write guarded CPU payloads into persistent input-slot spill storage."""

        self.bridge.wait_idle()
        actual = flat_runtime_arguments(self.capture, self.model, inputs)
        written: set[str] = set()
        for position, alias_id in self._user_alias_by_position.items():
            value = actual[position]
            if not isinstance(value, torch.Tensor):
                raise RuntimeError("captured tensor input became static")
            if value.device.type == "cuda":
                value = value.detach().cpu()
            if alias_id not in written:
                self.bridge.write_spill_tensor(alias_id, value)
                written.add(alias_id)
        for index, value in enumerate(actual):
            if not isinstance(value, torch.Tensor):
                self.root_arguments[index] = value
        return tuple(self.root_arguments)

    def prepare_invocation(self, inputs: Sequence[Any]) -> Sequence[object]:
        """Validate shared references and replace them with frontend shells."""

        leaves, tree_spec = tree_flatten(inputs)
        for item in self._shared_inputs:
            if item.root_input_index is None:
                raise AssertionError("shared input has no root position")
            try:
                supplied = leaves[item.public_leaf_index]
            except IndexError as error:
                raise InputGuardError(
                    f"shared input leaf {item.public_leaf_index} is missing"
                ) from error
            if isinstance(supplied, SharedInput):
                if supplied.require_in != item.require_in or (
                    supplied.consistency is not item.consistency
                ):
                    raise InputGuardError(
                        f"shared input leaf {item.public_leaf_index} changed its "
                        "pool or consistency declaration"
                    )
                reference = supplied.reference
            elif isinstance(supplied, TensorRef):
                reference = supplied
            else:
                raise InputGuardError(
                    f"shared input leaf {item.public_leaf_index} must be a "
                    "SharedInput or TensorRef"
                )
            self._validate_shared_reference(item, reference)
            shell = self.root_arguments[item.root_input_index]
            if not isinstance(shell, torch.Tensor):
                raise RuntimeError("shared input shell became static")
            leaves[item.public_leaf_index] = shell
        normalized = tree_unflatten(leaves, tree_spec)
        if not isinstance(normalized, tuple | list):
            raise InputGuardError("forward inputs must be a list or tuple")
        return normalized

    def _validate_shared_reference(
        self,
        expected: ResolvedSharedInput,
        actual: TensorRef,
    ) -> None:
        actual.object._require_open()
        if not actual.object._belongs_to(self.runtime):
            raise InputGuardError("shared input belongs to another Runtime")
        reference = expected.reference
        geometry = (
            actual.object.object_id,
            actual.object.size_bytes,
            actual.dtype,
            actual.shape,
            actual.stride,
            actual.storage_offset,
            actual.requires_grad,
        )
        expected_geometry = (
            reference.object.object_id,
            reference.object.size_bytes,
            reference.dtype,
            reference.shape,
            reference.stride,
            reference.storage_offset,
            reference.requires_grad,
        )
        if geometry != expected_geometry:
            raise InputGuardError(
                f"shared input leaf {expected.public_leaf_index} changed runtime "
                "identity or tensor geometry"
            )
        if expected.require_in not in actual.retained_pools:
            raise InputGuardError(
                f"shared input leaf {expected.public_leaf_index} no longer "
                f"guarantees pool {expected.require_in!r}"
            )

    def replacement_storage_views(self, alias_id: str) -> ReplacementStorageViews:
        """Collect persistent views before runtime task publication begins."""

        tensors: list[torch.Tensor] = []
        representative = self.object_store.get(alias_id)
        if representative is not None:
            tensors.append(representative)
        object_ids = set(self._object_ids_by_alias.get(alias_id, ()))
        for slot in self.lowered.root_input_slots:
            if slot.object_id in object_ids:
                value = self.root_arguments[slot.leaf_index]
                if isinstance(value, torch.Tensor):
                    tensors.append(value)
        for registration in self._registrations():
            if registration.binding.object_id in object_ids:
                tensors.append(registration.tensor)
        unique = unique_live_tensors(tensors)
        if not unique:
            raise RuntimeError(f"replacement alias {alias_id!r} has no frontend view")
        return ReplacementStorageViews(
            alias_id=alias_id,
            tensors=unique,
        )

    def publish_replacement_views(self, replacement: ReplacementStorageViews) -> None:
        """Keep the stable frontend representative rebound by the runtime boundary."""

        self.object_store[replacement.alias_id] = replacement.tensors[0]

    def restore_cpu_and_unregister(self) -> None:
        """Synchronize, write model state back, and reclaim all plan objects."""

        if self._closed:
            return
        self.bridge.wait_idle()
        registrations = self._registrations()
        restore_persistent_state(self.runtime, self._persistent_state)
        owners = self._read_model_aliases(
            aliases=self._registered_model_aliases - self._persistent_aliases
        )
        by_alias: dict[str, list[_Registration]] = {}
        for item in registrations:
            alias_id = self.bridge.alias_for_object(item.binding.object_id)
            if alias_id in self._persistent_aliases:
                continue
            if alias_id not in self._registered_model_aliases:
                continue
            by_alias.setdefault(alias_id, []).append(item)
        for alias_id, items in by_alias.items():
            owner = owners[alias_id]
            assigned: set[int] = set()
            for item in items:
                tensor = item.tensor
                if id(tensor) in assigned:
                    continue
                view = self._cpu_view(owner, tensor)
                tensor.data = view
                assigned.add(id(tensor))
        self.bridge.unregister(
            self.bridge.registered_aliases() - self._persistent_aliases
        )
        self.object_store.clear()
        self._closed = True

    def _empty_model_aliases(
        self,
        *,
        aliases: set[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        sizes = {
            group.alias_group_id: group.size_bytes
            for group in self.lowered.program.alias_groups
        }
        selected = self._registered_model_aliases if aliases is None else aliases
        for alias_id in selected:
            result[alias_id] = torch.empty(
                sizes[alias_id], dtype=torch.uint8, device="cpu"
            )
        return result

    def _read_model_aliases(
        self,
        *,
        aliases: set[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        self.bridge.wait_idle()
        owners = self._empty_model_aliases(aliases=aliases)
        for alias_id, owner in owners.items():
            self.bridge.read_spill_tensor(alias_id, owner)
        return owners

    @staticmethod
    def _cpu_view(owner: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
        return torch.empty(0, dtype=tensor.dtype, device="cpu").set_(
            owner.untyped_storage(),
            tensor.storage_offset(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
        )

    def _materialize(self) -> None:
        registrations = self._registrations()
        registration_by_id = {id(item.tensor): item for item in registrations}
        entries, shared_by_root = self._collect_materialization_entries(registrations)
        retain = {
            group.alias_group_id: group.retain_spill_copy
            for group in self.lowered.program.alias_groups
        }
        for ordinal, group in enumerate(self.lowered.program.alias_groups):
            values = entries.get(group.alias_group_id)
            if values:
                self._materialize_alias(
                    group.alias_group_id,
                    values,
                    registration_by_id,
                    shared_by_root,
                    retain_spill_copy=retain[group.alias_group_id],
                    ordinal=ordinal,
                )
        self.bridge.wait_idle()

    def _collect_materialization_entries(
        self,
        registrations: tuple[_Registration, ...],
    ) -> tuple[
        dict[str, list[_MaterializationEntry]],
        dict[int, ResolvedSharedInput],
    ]:
        entries: dict[str, list[_MaterializationEntry]] = {}
        for item in registrations:
            alias_id = self.bridge.alias_for_object(item.binding.object_id)
            self._model_aliases.add(alias_id)
            entries.setdefault(alias_id, []).append((item.tensor, None))
        slot_by_position = {
            slot.leaf_index: slot for slot in self.lowered.root_input_slots
        }
        shared_by_root = {
            item.root_input_index: item
            for item in self._shared_inputs
            if item.root_input_index is not None
        }
        for position, value in enumerate(self.root_arguments):
            slot = slot_by_position.get(position)
            if slot is None or not isinstance(value, torch.Tensor):
                continue
            alias_id = self.bridge.alias_for_object(slot.object_id)
            if alias_id not in self._model_aliases:
                if position in shared_by_root:
                    self._shared_alias_by_position[position] = alias_id
                else:
                    self._user_alias_by_position[position] = alias_id
                entries.setdefault(alias_id, []).append((value, position))
        return entries, shared_by_root

    def _materialize_alias(
        self,
        alias_id: str,
        values: list[_MaterializationEntry],
        registration_by_id: dict[int, _Registration],
        shared_by_root: dict[int, ResolvedSharedInput],
        *,
        retain_spill_copy: bool,
        ordinal: int,
    ) -> None:
        source = values[0][0]
        shared_items = self._shared_items(alias_id, values, shared_by_root)
        if shared_items:
            self._adopt_shared_alias(alias_id, shared_items)
        else:
            self._adopt_or_register_alias(
                alias_id,
                source,
                retain_spill_copy=retain_spill_copy,
            )
        if alias_id in self._model_aliases:
            self._registered_model_aliases.add(alias_id)
        owner, representative = self._build_device_views(
            values,
            registration_by_id,
        )
        if shared_items:
            torch.ops.shadowspill._dematerialize_storages([representative])
            self.object_store[alias_id] = representative
            return
        binding = self.bridge.publish_initial_tensor(alias_id, owner)
        self.object_store[alias_id] = representative
        task_number = (1 << 61) + ordinal
        actions = (
            MemoryAction("task_000000", alias_id, MemoryActionKind.RELEASE),
        )
        self.bridge.admit_initial_actions(actions, task_number=task_number)
        self.bridge.dematerialize(representative, alias_id, binding.generation)
        self.bridge.submit_initial_actions(
            actions,
            task_number=task_number,
        )

    def _shared_items(
        self,
        alias_id: str,
        values: list[_MaterializationEntry],
        shared_by_root: dict[int, ResolvedSharedInput],
    ) -> tuple[ResolvedSharedInput, ...]:
        items = tuple(
            shared_by_root[position]
            for position, candidate in self._shared_alias_by_position.items()
            if candidate == alias_id
        )
        if not items:
            return ()
        ordinary_positions = tuple(
            position
            for _value, position in values
            if position is not None and position not in shared_by_root
        )
        if ordinary_positions:
            raise PlanningError(
                "every public input view of a shared storage root must be "
                "declared as SharedInput; ordinary root positions are "
                f"{ordinary_positions}"
            )
        first = items[0]
        if any(
            item.reference.object.object_id != first.reference.object.object_id
            or item.consistency is not first.consistency
            for item in items[1:]
        ):
            raise PlanningError(
                "aliased shared input views disagree on object identity or consistency"
            )
        return items

    def _adopt_shared_alias(
        self,
        alias_id: str,
        items: tuple[ResolvedSharedInput, ...],
    ) -> None:
        first = items[0]
        self.bridge.adopt_shared_object(
            alias_id,
            first.reference.object,
            consistency=first.consistency,
        )

    def _adopt_or_register_alias(
        self,
        alias_id: str,
        source: torch.Tensor,
        *,
        retain_spill_copy: bool,
    ) -> None:
        persistent = adopt_persistent_tensor(
            self.runtime,
            self.model,
            source,
            self.bridge,
            alias_id,
        )
        if persistent is not None:
            self._persistent_aliases.add(alias_id)
            return
        if alias_id in self._model_aliases:
            raise PlanningError(
                f"registered model alias {alias_id!r} has no imported runtime storage"
            )
        self.bridge.register_spill_tensor(
            alias_id,
            source,
            retain_spill_copy=retain_spill_copy,
        )

    def _build_device_views(
        self,
        values: list[_MaterializationEntry],
        registration_by_id: dict[int, _Registration],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source = values[0][0]
        owner = torch.empty(
            source.untyped_storage().nbytes(), dtype=torch.uint8, device=self.device
        )
        views: dict[int, torch.Tensor] = {}
        for source_tensor, root_position in values:
            view = views.get(id(source_tensor))
            if view is None:
                view = torch.empty(0, dtype=source_tensor.dtype, device=self.device)
                view.set_(
                    owner.untyped_storage(),
                    source_tensor.storage_offset(),
                    tuple(source_tensor.shape),
                    tuple(source_tensor.stride()),
                )
                view.requires_grad_(source_tensor.requires_grad)
                if id(source_tensor) in registration_by_id:
                    source_tensor.data = view
                    view = source_tensor
                views[id(source_tensor)] = view
            if root_position is not None:
                self.root_arguments[root_position] = view
        return owner, next(iter(views.values()), owner)

    def _registrations(self) -> tuple[_Registration, ...]:
        values: list[_Registration] = []
        for binding in self.lowered.registrations:
            tensor = (
                self.model.get_parameter(binding.name)
                if binding.parameter
                else self.model.get_buffer(binding.name)
            )
            values.append(_Registration(binding, tensor))
        return tuple(values)


def retained_input_aliases(lowered: LoweredForwardProgram) -> frozenset[str]:
    """Return non-model root aliases whose spill storage is reused across calls."""

    registration_objects = {item.object_id for item in lowered.registrations}
    return frozenset(
        next(
            item.alias_group_id
            for item in lowered.program.objects
            if item.object_id == slot.object_id
        )
        for slot in lowered.root_input_slots
        if slot.object_id not in registration_objects
    )


__all__ = [
    "MaterializedForwardState",
    "flat_runtime_arguments",
    "representative_cpu_inputs",
    "retained_input_aliases",
]
