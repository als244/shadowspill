"""Frontend views rebound together when a task replaces one storage root."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class ReplacementStorageViews:
    alias_id: str
    tensors: tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        if not self.alias_id or not self.tensors:
            raise ValueError("replacement must name an alias and its frontend views")
