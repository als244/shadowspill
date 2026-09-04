"""Persistent PyTorch state records backed by generic runtime objects."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class TensorView:
    """One existing Tensor identity that views a persistent storage root."""

    tensor: torch.Tensor
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    requires_grad: bool


@dataclass(slots=True)
class PersistentStorage:
    """One authoritative runtime object and its frontend storage root."""

    persistent_object_id: int
    current_object_id: int
    pool_id: int
    size_bytes: int
    pool_pointer: int
    anchor: torch.Tensor
    views: tuple[TensorView, ...]
    frontend_storage_is_separate: bool

    @property
    def storage_identity(self) -> int:
        return int(self.anchor.untyped_storage()._cdata)


@dataclass(slots=True)
class PersistentState:
    """All persistent storage roots associated with one public Python object.

    ``owning_plan`` records who created this state, which is the whole of the
    lifetime rule: ``None`` means the caller imported it, so it outlives every
    plan and only the caller releases it; a plan handle means that plan
    created it, so closing that plan releases it. Nothing here knows whether
    the target is a model, an optimizer, or anything else.
    """

    target: object
    pool: str
    storages: tuple[PersistentStorage, ...]
    source_owner: object | None
    owning_plan: int | None = None

    def by_storage_identity(self) -> dict[int, PersistentStorage]:
        return {item.storage_identity: item for item in self.storages}


__all__ = ["PersistentState", "PersistentStorage", "TensorView"]
