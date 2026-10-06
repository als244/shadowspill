"""Resolve ordinary registered parameters to explicit replica/contribution groups."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from shadowspill.pytorch.representations import (
    component_at,
    is_wrapper,
    tensor_components,
)

from ._control import Control


@dataclass(frozen=True)
class Parameter:
    name: str
    aliases: tuple[str, ...]
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    requires_grad: bool
    replicas: tuple[int, ...]
    contributions: tuple[int, ...]
    components: tuple[tuple[tuple[str, ...], tuple[int, ...], str], ...] = ()

    def record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "aliases": list(self.aliases),
            "shape": list(self.shape),
            "stride": list(self.stride),
            "dtype": self.dtype,
            "requires_grad": self.requires_grad,
            "replicas": list(self.replicas),
            "contributions": list(self.contributions),
            **(
                {
                    "components": [
                        [list(path), list(shape), dtype]
                        for path, shape, dtype in self.components
                    ]
                }
                if self.components
                else {}
            ),
        }


def members(group: dist.ProcessGroup | None) -> tuple[int, ...]:
    return (
        (dist.get_rank(),)
        if group is None
        else tuple(dist.get_process_group_ranks(group))
    )


def validate_parameter_storage(model: nn.Module) -> None:
    """Reject ambiguous updates through distinct overlapping Parameters.

    Ordinary tied names reference one Parameter. Distinct slices of a flat bank
    are supported when their byte spans are disjoint; interleaved spans need
    explicit, separately owned storage for this first distributed contract.
    """
    storages: dict[int, list[tuple[int, int, str]]] = {}
    for name, parameter in (
        (name, component)
        for name, logical in model.named_parameters()
        for _, component in tensor_components(logical)
    ):
        if parameter.numel() == 0:
            continue
        key = parameter.untyped_storage()._cdata
        item = parameter.element_size()
        start = int(parameter.storage_offset()) * item
        extent = 1 + sum(
            (int(size) - 1) * int(stride)
            for size, stride in zip(parameter.shape, parameter.stride(), strict=True)
        )
        end = start + extent * item
        storages.setdefault(key, []).append((start, end, name))
    for values in storages.values():
        previous_end, previous_name = -1, ""
        for start, end, name in sorted(values):
            if start < previous_end and name != previous_name:
                raise ValueError(
                    f"{name} and {previous_name}: distinct Parameters have overlapping "
                    "storage spans; tie the same Parameter object "
                    "or use disjoint storage"
                )
            if end > previous_end:
                previous_end, previous_name = end, name


_INHERIT = object()


def resolve(
    model: nn.Module,
    *,
    replica_group: dist.ProcessGroup | None,
    replica_overrides: Iterable[
        tuple[Iterable[nn.Parameter | str], dist.ProcessGroup | None]
    ] = (),
    gradient_overrides: Iterable[
        tuple[Iterable[nn.Parameter | str], dist.ProcessGroup | None]
    ] = (),
    gradient_default: Any = _INHERIT,
) -> tuple[
    tuple[Parameter, ...],
    dict[str, tuple[dist.ProcessGroup | None, dist.ProcessGroup | None]],
]:
    """Overrides select registered parameters by object or exact registered name."""
    validate_parameter_storage(model)
    named = dict(model.named_parameters(remove_duplicate=False))
    names = {id(value): name for name, value in reversed(tuple(named.items()))}
    replicas = {name: replica_group for name in names.values()}
    gradients = (
        {}
        if gradient_default is _INHERIT
        else {name: gradient_default for name in names.values()}
    )

    def apply(
        overrides: Iterable[
            tuple[Iterable[nn.Parameter | str], dist.ProcessGroup | None]
        ],
        target: dict[str, dist.ProcessGroup | None],
    ) -> None:
        supplied: dict[str, tuple[int, ...]] = {}
        for parameters, group in overrides:
            if isinstance(parameters, (str, torch.Tensor)):
                raise TypeError(
                    "an override selects an iterable of parameter names or objects"
                )
            for parameter in parameters:
                value = (
                    named.get(parameter) if isinstance(parameter, str) else parameter
                )
                if id(value) not in names:
                    raise ValueError(
                        "ownership override names an unregistered parameter"
                    )
                name = names[id(value)]
                if name in supplied and supplied[name] != members(group):
                    raise ValueError(f"conflicting ownership overrides for {name}")
                supplied[name] = members(group)
                target[name] = group

    apply(replica_overrides, replicas)
    apply(gradient_overrides, gradients)
    groups = {}
    records = []
    for name, parameter in model.named_parameters():
        replica = replicas[name]
        gradient = gradients.get(name, replica)
        replica_members, gradient_members = members(replica), members(gradient)
        if (
            dist.get_rank() not in replica_members
            or dist.get_rank() not in gradient_members
        ):
            raise ValueError(f"{name}: this process must belong to its declared groups")
        if not set(gradient_members) <= set(replica_members):
            raise ValueError(
                f"{name}: gradient contributions must be within its replicas"
            )
        groups[name] = (replica, gradient)
        records.append(
            Parameter(
                name,
                tuple(alias for alias, value in named.items() if value is parameter),
                tuple(parameter.shape),
                tuple(parameter.stride()),
                str(parameter.dtype),
                parameter.requires_grad,
                replica_members,
                gradient_members,
                tuple(
                    (path, tuple(value.shape), str(value.dtype))
                    for path, value in tensor_components(parameter)
                )
                if is_wrapper(parameter)
                else (),
            )
        )
    return tuple(records), groups


def validate(
    control: Control, parameters: Sequence[Parameter]
) -> tuple[list[dict[str, Any]], ...]:
    descriptions = control.exchange(
        "parameter_ownership", [p.record() for p in parameters]
    )
    by_rank = {
        rank: {p["name"]: p for p in description}
        for rank, description in zip(control.members, descriptions, strict=True)
    }
    # Every participant performs the same validation, producing the same error.
    for rank, description in by_rank.items():
        for name, parameter in description.items():
            peers = parameter["replicas"]
            if not set(peers) <= set(control.members):
                raise ValueError(f"{name}: a replica is outside the participant group")
            signature = {k: v for k, v in parameter.items() if k != "contributions"}
            gradient_sets = []
            for peer in peers:
                other = by_rank[peer].get(name)
                if (
                    other is None
                    or {k: v for k, v in other.items() if k != "contributions"}
                    != signature
                ):
                    raise ValueError(
                        f"{name}: rank {rank} and replica {peer} "
                        "disagree on state signature"
                    )
                gradient_sets.append(frozenset(other["contributions"]))
            for left in gradient_sets:
                for right in gradient_sets:
                    if left & right and left != right:
                        raise ValueError(
                            f"{name}: contribution groups overlap without matching"
                        )
    return descriptions


def tiles(tensor: torch.Tensor, capacity_elements: int) -> Iterator[torch.Tensor]:
    if capacity_elements < 1:
        raise ValueError("initial-state chunk capacity must hold an element")
    if tensor.numel() == 0:
        return
    extents = [1] * tensor.ndim
    remaining = capacity_elements
    for axis in reversed(range(tensor.ndim)):
        extents[axis] = min(tensor.shape[axis], remaining)
        remaining //= extents[axis]
    starts = [
        range(0, size, extent)
        for size, extent in zip(tensor.shape, extents, strict=True)
    ]
    for origin in product(*starts):
        yield tensor[
            tuple(
                slice(start, start + extent)
                for start, extent in zip(origin, extents, strict=True)
            )
        ]


@torch.no_grad()
def synchronize_initial(
    model: nn.Module,
    parameters: Sequence[Parameter],
    control: Control,
    *,
    chunk_bytes: int = 16 << 20,
) -> None:
    """Synchronize replicas with bounded CPU tiles through the control group.

    This setup-only broadcast also works when the model's groups use NCCL.
    Participants outside a parameter's replicas discard the tile; no additional
    process groups or full-device model copies are needed. Buffers stay local.
    """
    descriptions = validate(control, parameters)
    unique = {}
    for description in descriptions:
        for item in description:
            if len(item["replicas"]) > 1:
                unique[(tuple(item["replicas"]), item["name"])] = item
    named = dict(model.named_parameters())
    for (replicas, name), item in sorted(unique.items()):
        components = item.get("components", (((), item["shape"], item["dtype"]),))
        for path, shape, dtype_name in components:
            dtype = getattr(torch, dtype_name.removeprefix("torch."))
            element_size = torch.empty((), dtype=dtype).element_size()
            geometry = torch.empty(tuple(shape), device="meta", dtype=dtype)
            target = (
                component_at(named[name], tuple(path))
                if dist.get_rank() in replicas
                else None
            )
            for index, piece in enumerate(tiles(geometry, chunk_bytes // element_size)):
                # Derive the matching strided slice from a storage offset in the
                # contiguous meta geometry; the live parameter may be noncontiguous.
                origin = []
                offset = int(piece.storage_offset())
                for stride in geometry.stride():
                    coordinate, offset = divmod(offset, stride)
                    origin.append(coordinate)
                slices = tuple(
                    slice(start, start + extent)
                    for start, extent in zip(origin, piece.shape, strict=True)
                )

                def broadcast(
                    piece: torch.Tensor = piece,
                    dtype: torch.dtype = dtype,
                    replicas: tuple[int, ...] = replicas,
                    target: torch.Tensor | None = target,
                    slices: tuple[slice, ...] = slices,
                ) -> None:
                    staging = torch.empty(tuple(piece.shape), dtype=dtype, device="cpu")
                    if dist.get_rank() == replicas[0]:
                        assert target is not None
                        staging.copy_(target[slices])
                    dist.broadcast(
                        staging.reshape(-1).view(torch.uint8),
                        src=replicas[0],
                        group=control.group,
                    )
                    if target is not None:
                        target[slices].copy_(staging)

                control.run(f"initialize/{replicas}/{name}/{path}/{index}", broadcast)
