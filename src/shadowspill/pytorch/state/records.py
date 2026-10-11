"""Persistent PyTorch state records backed by generic runtime objects."""

from __future__ import annotations

from dataclasses import dataclass, field

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
    # Metadata-only CPU storages retain shape/alias identity and reject value
    # access. Their authoritative bytes exist solely in the runtime pool.
    unbacked: bool = False

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

    ``holders`` are the admitted plans bound to this state, whose device
    placeholders its tensors point at while they live. Any of them runs on
    any holder's placeholders, so the state goes back to host views only when
    the last holder closes.
    """

    target: object
    pool: str
    storages: tuple[PersistentStorage, ...]
    source_owner: object | None
    owning_plan: int | None = None
    holders: set[int] = field(default_factory=set)

    _storage_index: dict[int, PersistentStorage] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        # Owners/anchors are finalized before PersistentState is registered.
        # Plan adoption rebinds the public views, never these storage anchors.
        self._storage_index = {item.storage_identity: item for item in self.storages}

    def by_storage_identity(self) -> dict[int, PersistentStorage]:
        return self._storage_index


__all__ = ["PersistentState", "PersistentStorage", "TensorView"]
