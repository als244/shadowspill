"""Persistent frontend views participating in a storage replacement."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from shadowspill.pytorch.runtime_adapter.bridge import RuntimeBridge


@dataclass(frozen=True, slots=True)
class ReplacementStorageViews:
    """Persistent frontend views rebound when one logical object is overwritten."""

    alias_id: str
    tensors: tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        if not self.alias_id:
            raise ValueError("replacement alias must be non-empty")
        if not self.tensors:
            raise ValueError("replacement must name at least one frontend view")


class MaterializedState:
    """What a materialized model's state does regardless of what it is for.

    Forward and training states differ in what they hold, not in how they
    read an alias back from spill, how they take a CPU view of one, or how
    they rebind a replaced view.
    """

    bridge: RuntimeBridge
    object_store: dict[str, torch.Tensor]

    def _empty_model_aliases(
        self, *, aliases: set[str] | None = None
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def publish_replacement_views(self, replacement: ReplacementStorageViews) -> None:
        """Keep the stable frontend representative rebound by the runtime boundary."""

        self.object_store[replacement.alias_id] = replacement.tensors[0]

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


__all__ = ["MaterializedState", "ReplacementStorageViews"]
