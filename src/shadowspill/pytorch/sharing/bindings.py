"""Resolve public sharing declarations against concrete pytrees."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.utils._pytree import (
    GetAttrKey,
    MappingKey,
    SequenceKey,
    tree_flatten,
    tree_flatten_with_path,
    tree_unflatten,
)

from shadowspill.pytorch.contracts import PlanningError
from shadowspill.runtime import ObjectConsistency

from .declarations import SharedInput, SharedOutput
from .paths import PytreePath, format_path, resolve_path
from .references import TensorRef


@dataclass(frozen=True, slots=True)
class ResolvedSharedInput:
    """One runtime-backed public input leaf and its plan-time requirements."""

    public_leaf_index: int
    reference: TensorRef
    require_in: str
    consistency: ObjectConsistency
    root_input_index: int | None = None


@dataclass(frozen=True, slots=True)
class ResolvedSharedOutput:
    """One declared public tensor leaf and its retained pool set."""

    public_leaf_index: int
    path: PytreePath
    retain_in: tuple[str, ...]


def resolve_shared_inputs(
    values: object,
    *,
    pool_names: Sequence[str],
    runtime: object,
) -> tuple[object, tuple[ResolvedSharedInput, ...]]:
    """Replace shared-reference leaves with deterministic CPU representatives."""

    leaves, tree_spec = tree_flatten(values)
    available = frozenset(pool_names)
    resolved: list[ResolvedSharedInput] = []
    owners: dict[int, torch.Tensor] = {}
    for index, value in enumerate(leaves):
        if not isinstance(value, SharedInput):
            continue
        reference = value.reference
        reference.object._require_open()
        if not reference.object._belongs_to(runtime):
            raise PlanningError(
                f"shared input leaf {index} belongs to another Runtime"
            )
        if value.require_in not in available:
            raise PlanningError(
                f"shared input leaf {index} requires unknown pool "
                f"{value.require_in!r}"
            )
        if value.require_in not in reference.retained_pools:
            raise PlanningError(
                f"shared input leaf {index} requires pool {value.require_in!r}, "
                "but its reference does not guarantee that residency"
            )
        owner = owners.get(reference.object.object_id)
        if owner is None:
            owner = torch.empty(reference.object.size_bytes, dtype=torch.uint8)
            owners[reference.object.object_id] = owner
        representative = torch.empty(0, dtype=reference.dtype).set_(
            owner.untyped_storage(),
            reference.storage_offset,
            reference.shape,
            reference.stride,
        )
        representative.requires_grad_(reference.requires_grad)
        _populate_shared_representative(
            representative,
            value.profiling_value,
            leaf_index=index,
        )
        leaves[index] = representative
        resolved.append(
            ResolvedSharedInput(
                public_leaf_index=index,
                reference=reference,
                require_in=value.require_in,
                consistency=value.consistency,
            )
        )
    return tree_unflatten(leaves, tree_spec), tuple(resolved)


def _populate_shared_representative(
    destination: torch.Tensor,
    supplied: torch.Tensor | None,
    *,
    leaf_index: int,
) -> None:
    with torch.no_grad():
        if supplied is not None:
            if supplied.device.type != "cpu":
                raise PlanningError(
                    f"shared input leaf {leaf_index} profiling value must be CPU"
                )
            if tuple(supplied.shape) != tuple(destination.shape) or (
                supplied.dtype != destination.dtype
            ):
                raise PlanningError(
                    f"shared input leaf {leaf_index} profiling value geometry differs"
                )
            destination.copy_(supplied)
            return
        if not destination.is_floating_point() and not destination.is_complex():
            raise PlanningError(
                f"shared input leaf {leaf_index} has control dtype "
                f"{destination.dtype}; provide profiling_value"
            )
        generator = torch.Generator(device="cpu").manual_seed(
            0x5348_0000 + leaf_index
        )
        if destination.is_floating_point():
            destination.copy_(
                torch.randn(
                    tuple(destination.shape),
                    dtype=torch.float32,
                    generator=generator,
                ).to(dtype=destination.dtype)
            )
            return
        real = torch.randn(
            tuple(destination.shape), dtype=torch.float32, generator=generator
        )
        imaginary = torch.randn(
            tuple(destination.shape), dtype=torch.float32, generator=generator
        )
        destination.copy_(torch.complex(real, imaginary).to(dtype=destination.dtype))


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


__all__ = [
    "ResolvedSharedInput",
    "ResolvedSharedOutput",
    "resolve_shared_inputs",
    "resolve_shared_outputs",
]
