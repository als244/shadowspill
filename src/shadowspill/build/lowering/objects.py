"""Canonical cross-task objects and alias bundles."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch

from shadowspill.errors import CaptureError
from shadowspill.ir import (
    AliasGroupSpec,
    ObjectRole,
    ObjectSpec,
    Persistence,
    SharedResidencyPolicy,
    TransferSpec,
)
from shadowspill.pytorch.accelerator import DEVICE_TYPE
from shadowspill.pytorch.capture.live_storage import (
    live_storage_bytes,
    live_storage_identity,
)

from .geometry import _view_extent_bytes


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
        self._alias_transfers: dict[str, TransferSpec] = {}
        self._retain_spill: set[str] = set()
        self._shared_residency: dict[str, SharedResidencyPolicy] = {}
        self._object_by_key: dict[_TensorKey, str] = {}
        self._objects: list[_ObjectRecord] = []
        self._object_specs: tuple[ObjectSpec, ...] | None = None
        self._next_alias_id = 0

    def tensor_geometry(self, object_id: str) -> dict[str, object]:
        """Layout of an explicitly registered tensor, independent of its readers."""
        for key, identity in self._object_by_key.items():
            if identity == object_id:
                return {
                    "dtype": str(key.dtype),
                    "element_size": key.dtype.itemsize,
                    "shape": key.shape,
                    "stride": key.stride,
                    "storage_offset": key.storage_offset,
                }
        raise ValueError(f"object {object_id!r} has no registered tensor geometry")

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
        *,
        alias_id: str,
        offset_bytes: int,
        span_bytes: int,
        role: ObjectRole,
        persistence: Persistence,
        tensor: torch.Tensor | None = None,
    ) -> str:
        """Create one graph-declared view in an existing residency bundle.

        The contract says where the view sits and how far it reaches; a
        tensor, where one was returned, is kept alive because the storage
        it names has to outlive the call that produced it.
        """

        self._require_within(alias_id, offset_bytes, span_bytes, "task output")
        if tensor is not None:
            self._tensor_keepalive.append(tensor)
        return self._new_record(
            alias_id=alias_id,
            offset_bytes=offset_bytes,
            size_bytes=span_bytes,
            role=role,
            persistence=persistence,
        )

    def pad_alias(self, alias_id: str, size_bytes: int) -> None:
        """Extend a residency bundle for equal shards, preserving existing views."""
        if alias_id not in self._alias_sizes:
            raise CaptureError(f"unknown storage alias: {alias_id}")
        if size_bytes < self._alias_sizes[alias_id]:
            raise CaptureError("padding cannot shrink an existing storage alias")
        self._alias_sizes[alias_id] = size_bytes

    def validate_canonical_output_view(
        self,
        object_id: str,
        *,
        alias_id: str,
        offset_bytes: int,
        span_bytes: int,
        tensor: torch.Tensor | None = None,
    ) -> None:
        """Validate one compiled output against an existing canonical object.

        This deliberately never merges alias groups.  Compiled input reuse is
        a task-local lease handoff, not proof that two cross-task objects share
        one permanent residency bundle.
        """

        if tensor is not None:
            self._tensor_keepalive.append(tensor)
        record = self._record(object_id)
        if record.alias_group_id != alias_id:
            raise CaptureError("canonical output changed its alias group")
        self._require_within(alias_id, offset_bytes, span_bytes, "canonical output")
        if record.offset_bytes != offset_bytes:
            raise CaptureError(
                "canonical output changed its storage offset: "
                f"object={object_id}, expected={record.offset_bytes}, "
                f"actual={offset_bytes}"
            )
        if record.size_bytes != span_bytes:
            raise CaptureError(
                "canonical output changed its byte span: "
                f"object={object_id}, expected={record.size_bytes}, "
                f"actual={span_bytes}"
            )

    def _require_within(
        self,
        alias_id: str,
        offset_bytes: int,
        span_bytes: int,
        what: str,
    ) -> None:
        extent = self._alias_sizes.get(alias_id)
        if extent is None:
            raise CaptureError(f"{what} references an unknown alias group")
        if offset_bytes + span_bytes > extent:
            raise CaptureError(
                f"{what} view exceeds its declared storage: "
                f"alias_group={alias_id}, span=[{offset_bytes}, "
                f"{offset_bytes + span_bytes}), extent={extent}"
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
        storage_bytes = (
            live_storage_bytes(tensor) if tensor.device.type == DEVICE_TYPE else 0
        )
        if alias_id is None:
            alias_id = self._new_alias(storage_bytes)
            self._alias_by_storage[key.storage_identity] = alias_id
        elif self._alias_sizes[alias_id] != storage_bytes:
            raise CaptureError("one capture storage reported inconsistent byte extents")
        if retain_spill_copy:
            self._retain_spill.add(alias_id)
        return self._new_object(
            key,
            alias_id=alias_id,
            offset_bytes=(
                int(tensor.storage_offset()) * tensor.element_size()
                if tensor.device.type == DEVICE_TYPE
                else 0
            ),
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
    ) -> str:
        object_id = self._new_record(
            alias_id=alias_id,
            offset_bytes=offset_bytes,
            size_bytes=_view_extent_bytes(tensor),
            role=role,
            persistence=persistence,
        )
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
            self._retain_spill.add(record.alias_group_id)
        if changed:
            self._object_specs = None

    def alias_id(self, object_id: str) -> str:
        return self._record(object_id).alias_group_id

    def mark_output(self, object_id: str) -> None:
        record = self._record(object_id)
        if record.role in {
            ObjectRole.OTHER,
            ObjectRole.ACTIVATION,
            ObjectRole.CONTROL,
        }:
            record.role = ObjectRole.OUTPUT
            self._object_specs = None

    def mark_shared_residency(
        self,
        object_id: str,
        policy: SharedResidencyPolicy,
        *,
        retain_spill_copy: bool,
    ) -> None:
        """Declare one externally owned alias as execution-resident."""

        record = self._record(object_id)
        existing = self._shared_residency.get(record.alias_group_id)
        if existing is not None and existing is not policy:
            raise CaptureError(
                "aliased shared inputs declare different consistency policies"
            )
        self._shared_residency[record.alias_group_id] = policy
        if retain_spill_copy:
            self._retain_spill.add(record.alias_group_id)

    def finalize_shared_writes(self, written_object_ids: Iterable[str]) -> None:
        """Downgrade proven non-writing shared roots to read-only residency."""

        written_aliases = {
            self._record(object_id).alias_group_id for object_id in written_object_ids
        }
        for alias_id, policy in tuple(self._shared_residency.items()):
            if alias_id in written_aliases or (
                policy is SharedResidencyPolicy.SHARED_READ_ONLY
            ):
                continue
            self._shared_residency[alias_id] = SharedResidencyPolicy.SHARED_READ_ONLY

    def _record(self, object_id: str) -> _ObjectRecord:
        index = int(object_id.removeprefix("object_"))
        return self._objects[index]

    def alias_groups(self) -> tuple[AliasGroupSpec, ...]:
        return tuple(
            AliasGroupSpec(
                alias_group_id,
                self._device_id,
                size_bytes,
                retain_spill_copy=alias_group_id in self._retain_spill,
                shared_residency=self._shared_residency.get(alias_group_id),
                transfer=self._alias_transfers.get(alias_group_id),
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
