"""Persistent frontend views participating in a storage replacement."""

from __future__ import annotations

from dataclasses import dataclass

import torch


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


__all__ = ["ReplacementStorageViews"]
