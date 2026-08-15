"""Canonical cross-task objects and alias bundles."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from shadowspill.ir import AliasGroupSpec, ObjectRole, ObjectSpec, Persistence
from shadowspill.pytorch.capture.live_storage import (
    live_storage_bytes,
    live_storage_identity,
    live_view_key,
)
from shadowspill.pytorch.capture.storage import OutputView

from ..contracts import CaptureError


@dataclass(frozen=True, slots=True)
class TensorSlot:
    """Position of one tensor leaf in a framework task ABI."""

    leaf_index: int
    object_id: str


@dataclass(frozen=True, slots=True)
class RegistrationBinding:
    """Original registered name associated with one logical tensor view."""

    name: str
    object_id: str
    parameter: bool


@dataclass(frozen=True, slots=True)
class _TensorKey:
    storage_identity: int
    storage_offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype


@dataclass(slots=True)
class _ObjectRecord:
    object_id: str
    alias_group_id: str
    offset_bytes: int
    size_bytes: int
    role: ObjectRole
    persistence: Persistence


_ROLE_PRIORITY = {
    ObjectRole.OTHER: 0,
    ObjectRole.INPUT: 1,
    ObjectRole.ACTIVATION: 2,
    ObjectRole.CONTROL: 2,
    ObjectRole.OUTPUT: 3,
    ObjectRole.GRADIENT: 3,
    ObjectRole.OPTIMIZER_STATE: 4,
    ObjectRole.BUFFER: 5,
    ObjectRole.PARAMETER: 6,
}


def tensor_value_role(
    tensor: torch.Tensor,
    *,
    continuous_role: ObjectRole,
) -> ObjectRole:
    """Classify integer/boolean values as control objects."""

    return (
        continuous_role
        if tensor.is_floating_point() or tensor.is_complex()
        else ObjectRole.CONTROL
    )


def serialized_dtype_role(
    dtype_name: str,
    *,
    continuous_role: ObjectRole,
) -> ObjectRole:
    """Classify one storage-contract dtype without materializing a tensor."""

    name = dtype_name.removeprefix("torch.")
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise CaptureError(f"storage contract has unknown dtype {dtype_name!r}")
    return (
        continuous_role
        if dtype.is_floating_point or dtype.is_complex
        else ObjectRole.CONTROL
    )


def _view_extent_bytes(tensor: torch.Tensor) -> int:
    """Return the compact storage span needed for one strided tensor view."""

    if tensor.numel() == 0:
        return 0
    if any(stride < 0 for stride in tensor.stride()):
        raise CaptureError("compiled task output has a negative stride")
    last_element = sum(
        (extent - 1) * stride
        for extent, stride in zip(tensor.shape, tensor.stride(), strict=True)
    )
    return int((last_element + 1) * tensor.element_size())


def _contract_view_size(view: OutputView) -> int:
    """Return logical bytes from one already-validated symbolic output view."""

    elements = math.prod(view.shape)
    if elements == 0:
        return 0
    span_elements = 1 + sum(
        (extent - 1) * stride
        for extent, stride in zip(view.shape, view.stride, strict=True)
    )
    if span_elements <= 0 or view.span_bytes % span_elements:
        raise CaptureError("task output view has an invalid symbolic byte span")
    return elements * (view.span_bytes // span_elements)


class ObjectCatalog:
    """Canonical cross-task objects and alias bundles.

    Live framework storage identities are accepted only by ``add`` for
    user-owned inputs, parameters, buffers, and their views. Task outputs enter
    through ``TaskBindingResolver`` using the offline semantic contract and
    reconciled compiled layout.
    """

    def __init__(self, *, device_id: str) -> None:
        self._device_id = device_id
        self._tensor_keepalive: list[torch.Tensor] = []
        self._alias_by_storage: dict[int, str] = {}
        self._alias_sizes: dict[str, int] = {}
        self._retain_host: set[str] = set()
        self._object_by_key: dict[_TensorKey, str] = {}
        self._objects: list[_ObjectRecord] = []
        self._object_specs: tuple[ObjectSpec, ...] | None = None
        self._next_alias_id = 0

    @staticmethod
    def key(tensor: torch.Tensor) -> _TensorKey:
        if tensor.layout is not torch.strided:
            raise CaptureError("program lowering currently requires strided tensors")
        return _TensorKey(
            storage_identity=live_storage_identity(tensor),
            storage_offset=int(tensor.storage_offset()),
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
            dtype=tensor.dtype,
        )

    def add(
        self,
        tensor: torch.Tensor,
        *,
        role: ObjectRole,
        persistence: Persistence,
        retain_spill_copy: bool = False,
    ) -> str:
        return self._add(
            tensor,
            role=role,
            persistence=persistence,
            retain_spill_copy=retain_spill_copy,
        )

    def add_output_view(
        self,
        tensor: torch.Tensor,
        *,
        alias_id: str,
        offset_bytes: int,
        role: ObjectRole,
        persistence: Persistence,
    ) -> str:
        """Create one graph-declared view in an existing residency bundle."""

        if alias_id not in self._alias_sizes:
            raise CaptureError("task output references an unknown alias group")
        if offset_bytes + _view_extent_bytes(tensor) > self._alias_sizes[alias_id]:
            raise CaptureError("task output view exceeds its declared storage")
        self._tensor_keepalive.append(tensor)
        return self._new_object(
            self.key(tensor),
            alias_id=alias_id,
            offset_bytes=offset_bytes,
            tensor=tensor,
            role=role,
            persistence=persistence,
            index_by_tensor_key=False,
        )

    def add_contract_output_view(
        self,
        view: OutputView,
        *,
        alias_id: str,
        offset_bytes: int,
        role: ObjectRole,
        persistence: Persistence,
    ) -> str:
        """Create an output object directly from the offline task contract."""

        if alias_id not in self._alias_sizes:
            raise CaptureError("task output references an unknown alias group")
        if offset_bytes + view.span_bytes > self._alias_sizes[alias_id]:
            raise CaptureError("task output view exceeds its declared storage")
        return self._new_record(
            alias_id=alias_id,
            offset_bytes=offset_bytes,
            size_bytes=_contract_view_size(view),
            role=role,
            persistence=persistence,
        )

    def validate_canonical_output_view(
        self,
        tensor: torch.Tensor,
        object_id: str,
        *,
        alias_id: str,
        offset_bytes: int,
    ) -> None:
        """Validate a compiled output against an existing canonical object.

        This deliberately never merges alias groups.  Compiled input reuse is
        a task-local lease handoff, not proof that two cross-task objects share
        one permanent residency bundle.
        """

        self._tensor_keepalive.append(tensor)
        record = self._record(object_id)
        if record.alias_group_id != alias_id:
            raise CaptureError("canonical output changed its alias group")
        if offset_bytes + _view_extent_bytes(tensor) > self._alias_sizes[alias_id]:
            raise CaptureError("canonical output exceeds its declared storage")
        if record.offset_bytes != offset_bytes:
            raise CaptureError(
                "canonical output changed its storage offset: "
                f"object={object_id}, expected={record.offset_bytes}, "
                f"actual={offset_bytes}"
            )
        size_bytes = int(tensor.numel()) * tensor.element_size()
        if record.size_bytes != size_bytes:
            raise CaptureError(
                "canonical output changed its logical byte size: "
                f"object={object_id}, expected={record.size_bytes}, "
                f"actual={size_bytes}"
            )

    def validate_canonical_contract_view(
        self,
        view: OutputView,
        object_id: str,
        *,
        alias_id: str,
        offset_bytes: int,
    ) -> None:
        """Validate a contract view against an existing canonical object."""

        record = self._record(object_id)
        if record.alias_group_id != alias_id:
            raise CaptureError("canonical output changed its alias group")
        if offset_bytes + view.span_bytes > self._alias_sizes[alias_id]:
            raise CaptureError("canonical output exceeds its declared storage")
        if record.offset_bytes != offset_bytes:
            raise CaptureError(
                "canonical output changed its storage offset: "
                f"object={object_id}, expected={record.offset_bytes}, "
                f"actual={offset_bytes}"
            )
        size_bytes = _contract_view_size(view)
        if record.size_bytes != size_bytes:
            raise CaptureError(
                "canonical output changed its logical byte size: "
                f"object={object_id}, expected={record.size_bytes}, "
                f"actual={size_bytes}"
            )

    def _add(
        self,
        tensor: torch.Tensor,
        *,
        role: ObjectRole,
        persistence: Persistence,
        retain_spill_copy: bool,
    ) -> str:
        self._tensor_keepalive.append(tensor)
        key = self.key(tensor)
        object_id = self._object_by_key.get(key)
        if object_id is not None:
            self._upgrade_object(
                object_id,
                role=role,
                persistence=persistence,
                retain_spill_copy=retain_spill_copy,
            )
            return object_id
        alias_id = self._alias_by_storage.get(key.storage_identity)
        storage_bytes = live_storage_bytes(tensor)
        if alias_id is None:
            alias_id = self._new_alias(storage_bytes)
            self._alias_by_storage[key.storage_identity] = alias_id
        elif self._alias_sizes[alias_id] != storage_bytes:
            raise CaptureError("one capture storage reported inconsistent byte extents")
        if retain_spill_copy:
            self._retain_host.add(alias_id)
        return self._new_object(
            key,
            alias_id=alias_id,
            offset_bytes=int(tensor.storage_offset()) * tensor.element_size(),
            tensor=tensor,
            role=role,
            persistence=persistence,
        )

    def _new_alias(self, size_bytes: int) -> str:
        alias_id = f"alias_{self._next_alias_id:06d}"
        self._next_alias_id += 1
        self._alias_sizes[alias_id] = size_bytes
        return alias_id

    def new_output_alias(self, size_bytes: int) -> str:
        """Create one task-local storage bundle declared by the FX graph."""

        return self._new_alias(size_bytes)

    def alias_size(self, alias_id: str) -> int:
        """Return one residency bundle's declared physical extent."""

        try:
            return self._alias_sizes[alias_id]
        except KeyError as exc:
            raise CaptureError("task references an unknown alias group") from exc

    def _new_object(
        self,
        key: _TensorKey,
        *,
        alias_id: str,
        offset_bytes: int,
        tensor: torch.Tensor,
        role: ObjectRole,
        persistence: Persistence,
        index_by_tensor_key: bool = True,
    ) -> str:
        object_id = self._new_record(
            alias_id=alias_id,
            offset_bytes=offset_bytes,
            size_bytes=int(tensor.numel()) * tensor.element_size(),
            role=role,
            persistence=persistence,
        )
        if index_by_tensor_key:
            self._object_by_key[key] = object_id
        return object_id

    def _new_record(
        self,
        *,
        alias_id: str,
        offset_bytes: int,
        size_bytes: int,
        role: ObjectRole,
        persistence: Persistence,
    ) -> str:
        object_id = f"object_{len(self._objects):06d}"
        self._objects.append(
            _ObjectRecord(
                object_id=object_id,
                alias_group_id=alias_id,
                offset_bytes=offset_bytes,
                size_bytes=size_bytes,
                role=role,
                persistence=persistence,
            )
        )
        self._object_specs = None
        return object_id

    def _upgrade_object(
        self,
        object_id: str,
        *,
        role: ObjectRole,
        persistence: Persistence,
        retain_spill_copy: bool,
    ) -> None:
        record = self._record(object_id)
        changed = False
        if (
            persistence is Persistence.CHECKPOINT
            and record.persistence is not Persistence.CHECKPOINT
        ):
            record.persistence = persistence
            changed = True
        if _ROLE_PRIORITY[role] > _ROLE_PRIORITY[record.role]:
            record.role = role
            changed = True
        if retain_spill_copy:
            self._retain_host.add(record.alias_group_id)
        if changed:
            self._object_specs = None

    def alias_id(self, object_id: str) -> str:
        return self._record(object_id).alias_group_id

    def object_size(self, object_id: str) -> int:
        return self._record(object_id).size_bytes

    def mark_output(self, object_id: str) -> None:
        record = self._record(object_id)
        if record.role in {
            ObjectRole.OTHER,
            ObjectRole.ACTIVATION,
            ObjectRole.CONTROL,
        }:
            record.role = ObjectRole.OUTPUT
            self._object_specs = None

    def _record(self, object_id: str) -> _ObjectRecord:
        index = int(object_id.removeprefix("object_"))
        return self._objects[index]

    def alias_groups(self) -> tuple[AliasGroupSpec, ...]:
        return tuple(
            AliasGroupSpec(
                alias_group_id,
                self._device_id,
                size_bytes,
                retain_spill_copy=alias_group_id in self._retain_host,
            )
            for alias_group_id, size_bytes in self._alias_sizes.items()
        )

    def objects(self) -> tuple[ObjectSpec, ...]:
        if self._object_specs is None:
            self._object_specs = tuple(
                ObjectSpec(
                    item.object_id,
                    item.alias_group_id,
                    item.offset_bytes,
                    item.size_bytes,
                    item.role,
                    item.persistence,
                )
                for item in self._objects
            )
        return self._object_specs


def register_model_state(
    model: nn.Module,
    catalog: ObjectCatalog,
) -> tuple[tuple[RegistrationBinding, ...], dict[tuple[int, int], str]]:
    """Register parameters and buffers once for every lowering mode."""

    registrations: list[RegistrationBinding] = []
    parameter_objects: dict[tuple[int, int], str] = {}
    checkpoint_names = set(model.state_dict())
    for name, parameter in model.named_parameters(remove_duplicate=False):
        object_id = catalog.add(
            parameter,
            role=ObjectRole.PARAMETER,
            persistence=Persistence.CHECKPOINT,
            retain_spill_copy=True,
        )
        registrations.append(RegistrationBinding(name, object_id, True))
        parameter_objects[live_view_key(parameter)] = object_id
    for name, buffer in model.named_buffers(remove_duplicate=False):
        object_id = catalog.add(
            buffer,
            role=ObjectRole.BUFFER,
            persistence=(
                Persistence.CHECKPOINT if name in checkpoint_names else Persistence.RUN
            ),
            retain_spill_copy=True,
        )
        registrations.append(RegistrationBinding(name, object_id, False))
    return tuple(registrations), parameter_objects


__all__ = [
    "ObjectCatalog",
    "RegistrationBinding",
    "TensorSlot",
    "register_model_state",
    "serialized_dtype_role",
    "tensor_value_role",
]
