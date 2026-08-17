"""Resolve public sharing declarations against concrete pytrees."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.utils._pytree import (
    GetAttrKey,
    MappingKey,
    SequenceKey,
    tree_flatten_with_path,
)

from .declarations import SharedOutput
from .paths import PytreePath, format_path, resolve_path


@dataclass(frozen=True, slots=True)
class ResolvedSharedOutput:
    """One declared public tensor leaf and its retained pool set."""

    public_leaf_index: int
    path: PytreePath
    retain_in: tuple[str, ...]


def resolve_shared_outputs(
    output: object,
    declarations: Sequence[SharedOutput],
    *,
    pool_names: Sequence[str],
) -> tuple[ResolvedSharedOutput, ...]:
    """Resolve declarations by path without relying on tensor identity."""

    normalized = tuple(declarations)
    if any(not isinstance(item, SharedOutput) for item in normalized):
        raise TypeError("shared_outputs must contain SharedOutput declarations")
    available = frozenset(pool_names)
    leaves, _ = tree_flatten_with_path(output)
    leaf_by_path = {
        _public_path(key_path): (index, leaf)
        for index, (key_path, leaf) in enumerate(leaves)
    }
    resolved: list[ResolvedSharedOutput] = []
    seen: set[int] = set()
    for declaration in normalized:
        retained_pools = (
            (declaration.retain_in,)
            if isinstance(declaration.retain_in, str)
            else declaration.retain_in
        )
        unknown = set(retained_pools) - available
        if unknown:
            raise ValueError(
                f"shared output {format_path(declaration.path)} names unknown "
                f"runtime pools {sorted(unknown)}"
            )
        # Resolve first for precise missing/intermediate-component diagnostics.
        value = resolve_path(output, declaration.path)
        match = leaf_by_path.get(declaration.path)
        if match is None or match[1] is not value:
            raise ValueError(
                f"shared output {format_path(declaration.path)} must identify "
                "one public pytree leaf"
            )
        leaf_index, leaf = match
        if not isinstance(leaf, torch.Tensor):
            raise TypeError(
                f"shared output {format_path(declaration.path)} is not a tensor"
            )
        if leaf_index in seen:
            raise ValueError(
                f"shared output {format_path(declaration.path)} is declared twice"
            )
        seen.add(leaf_index)
        resolved.append(
            ResolvedSharedOutput(
                public_leaf_index=leaf_index,
                path=declaration.path,
                retain_in=retained_pools,
            )
        )
    return tuple(sorted(resolved, key=lambda item: item.public_leaf_index))


def _public_path(keys: tuple[object, ...]) -> PytreePath:
    result: list[str | int] = []
    for key in keys:
        if isinstance(key, SequenceKey):
            result.append(int(key.idx))
        elif isinstance(key, GetAttrKey):
            result.append(str(key.name))
        elif isinstance(key, MappingKey) and isinstance(key.key, str):
            result.append(key.key)
        else:
            raise TypeError(
                "shared-output paths support sequences, attributes, and "
                "string-keyed mappings"
            )
    return tuple(result)


__all__ = ["ResolvedSharedOutput", "resolve_shared_outputs"]
