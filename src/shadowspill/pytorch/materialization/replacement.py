"""Persistent frontend views participating in a storage replacement."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class ReplacementStorageViews:
    """One logical object's views before a task publishes its next generation."""

    alias_id: str
    previous_generation: int
    tensors: tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        if not self.alias_id:
            raise ValueError("replacement alias must be non-empty")
        if self.previous_generation < 0:
            raise ValueError("replacement generation must be non-negative")
        if not self.tensors:
            raise ValueError("replacement must name at least one frontend view")


__all__ = ["ReplacementStorageViews"]
