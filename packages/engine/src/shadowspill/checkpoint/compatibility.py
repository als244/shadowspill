"""Global logical identity and structural ties before checkpoint conversion."""

from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from math import gcd, prod
from types import MappingProxyType
from typing import Any

from shadowspill.artifacts._values import digest
from shadowspill.artifacts.state import StateId, StateRole

from .logical import LogicalTensor
from .mapping import coverage
from .state import PlanState


@dataclass(frozen=True)
class RankedTensor:
    rank: int
    tensor: LogicalTensor


@dataclass(frozen=True)
class LogicalFamily:
    tensors: Mapping[StateId, tuple[RankedTensor, ...]]
    identities: Mapping[StateId, tuple[Any, ...]]
    ties: Mapping[StateId, frozenset[StateId]]
    relations: Mapping[StateId, tuple[Any, ...]]

    @property
    def identity(self) -> str:
        return digest(
            [
                {
                    "state": key.to_dict(),
                    "semantics": self.identities[key],
                    "ties": [other.to_dict() for other in sorted(self.ties[key])],
                    "view_relation": self.relations[key],
                }
                for key in sorted(self.tensors)
            ]
        )


def describe_family(
    states: Sequence[PlanState], *, roles: Collection[StateRole] | None = None
) -> LogicalFamily:
    """Check the complete rank set, global coordinates and replica descriptions."""
    if not states or [state.logical_rank for state in states] != list(
        range(len(states))
    ):
        raise ValueError(
            "checkpoint conversion requires each rank exactly once in order"
        )
    tensors: dict[StateId, list[RankedTensor]] = defaultdict(list)
    identities: dict[StateId, tuple[Any, ...]] = {}
    ties: dict[StateId, frozenset[StateId]] = {}
    relations: dict[StateId, tuple[Any, ...]] = {}
    for state in states:
        for plan in (state.plan, *state.plans.values()):
            count = (
                1
                if plan.program.parallelism is None
                else plan.program.parallelism.world_size
            )
            if count != len(states):
                raise ValueError("checkpoint rank set differs from saved Parallelism")
        groups: dict[str | StateId, dict[StateId, LogicalTensor]] = defaultdict(dict)
        for item in state.logical_tensors(roles=roles):
            _nonoverlapping(item)
            key = item.tensor.identity
            signature: tuple[Any, ...] = (
                item.view.shape,
                item.tensor.geometry["dtype"],
                item.tensor.parameter_name,
                item.tensor.optimizer_type,
                item.tensor.ownership.get("trainable"),
            )
            if component := item.tensor.ownership.get("component_layout"):
                signature = (
                    *signature,
                    component["encoding"],
                    tuple(component["axes"]),
                )
            if identities.setdefault(key, signature) != signature:
                raise ValueError(f"checkpoint ranks disagree on logical state: {key}")
            tensors[key].append(RankedTensor(state.logical_rank, item))
            groups[item.archive.name if item.archive is not None else key][key] = item
        for group in groups.values():
            members = frozenset(group)
            anchor = group[min(group)]
            for key in members:
                if ties.setdefault(key, members) != members:
                    raise ValueError(
                        f"checkpoint ranks disagree on structural ties: {key}"
                    )
                relation = _relation(anchor, group[key])
                if relations.setdefault(key, relation) != relation:
                    raise ValueError(f"checkpoint ranks disagree on tied views: {key}")
    for entries in tensors.values():
        for _ in coverage([entry.tensor.view for entry in entries]):
            pass
    return LogicalFamily(
        MappingProxyType({key: tuple(entries) for key, entries in tensors.items()}),
        MappingProxyType(identities),
        MappingProxyType(ties),
        MappingProxyType(relations),
    )


def require_compatible(source: LogicalFamily, target: LogicalFamily) -> None:
    """Never invent/reset state or infer ties from coincidentally equal values."""
    if source.tensors.keys() != target.tensors.keys():
        raise ValueError(
            "checkpoint logical inventory differs; explicit conversion is required"
        )
    for key in source.tensors:
        if source.identities[key] != target.identities[key]:
            raise ValueError(f"checkpoint logical state interpretation differs: {key}")
        if source.ties[key] != target.ties[key]:
            raise ValueError(f"checkpoint structural ties differ: {key}")
        if source.relations[key] != target.relations[key]:
            raise ValueError(f"checkpoint tied view relationship differs: {key}")


def _nonoverlapping(item: LogicalTensor) -> None:
    """Prove independent local coordinates; unknown overlaps need a declaration."""
    view = item.view
    if not prod(view.local_shape):
        return
    extent = 1
    for stride, size in sorted(zip(view.stride, view.local_shape, strict=True)):
        if size <= 1:
            continue
        if stride < extent:
            raise ValueError(
                "overlapping state views require an explicit conversion map"
            )
        extent += stride * (size - 1)


def _relation(anchor: LogicalTensor, item: LogicalTensor) -> tuple[Any, ...]:
    """Preserve coordinate relationships, not only alias membership or values.

    Axis permutations survive different model slices and physical storage order.
    More general tied views retain their explicit relative geometry; changing
    that geometry needs a declared transformation, not an equal-value guess.
    """
    left, right = anchor.view, item.view
    if anchor.tensor.identity == item.tensor.identity or all(
        getattr(left, name) == getattr(right, name)
        for name in ("shape", "local_shape", "origin", "stride", "storage_offset")
    ):
        return ("identity",)
    if not prod(left.shape) and not prod(right.shape):
        return ("empty",)
    axes: list[int | None] = []
    if left.storage_offset == right.storage_offset:
        for index, size in enumerate(right.shape):
            if size == 1:
                axes.append(None)
                continue
            matches = [
                axis
                for axis, extent in enumerate(left.shape)
                if axis not in axes
                and extent == size
                and left.local_shape[axis] == right.local_shape[index]
                and left.origin[axis] == right.origin[index]
                and left.stride[axis] == right.stride[index]
            ]
            if len(matches) != 1:
                break
            axes.append(matches[0])
        else:
            if {axis for axis in axes if axis is not None} == {
                axis for axis, size in enumerate(left.shape) if size != 1
            }:
                return "axes", tuple(axes)
    delta = right.storage_offset - left.storage_offset
    left_stride = tuple(
        step if size > 1 else 0
        for step, size in zip(left.stride, left.local_shape, strict=True)
    )
    right_stride = tuple(
        step if size > 1 else 0
        for step, size in zip(right.stride, right.local_shape, strict=True)
    )
    scale = gcd(*left_stride, *right_stride, abs(delta)) or 1
    return (
        "relative_geometry",
        left.shape,
        left.local_shape,
        left.origin,
        tuple(step // scale for step in left_stride),
        right.shape,
        right.local_shape,
        right.origin,
        tuple(step // scale for step in right_stride),
        delta // scale,
    )
