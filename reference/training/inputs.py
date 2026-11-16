"""Move ordinary reference inputs while preserving repeated tensor identities."""

from typing import Any

import torch
from torch.utils._pytree import tree_map


def on_device(value: Any, device: torch.device) -> Any:
    # Preserve repeated tensor identities in arbitrary nested inputs.
    tensors: dict[int, torch.Tensor] = {}

    def move(leaf: Any) -> Any:
        if not isinstance(leaf, torch.Tensor):
            return leaf
        if id(leaf) not in tensors:
            tensors[id(leaf)] = leaf.to(device=device)
        return tensors[id(leaf)]

    return tree_map(move, value)
