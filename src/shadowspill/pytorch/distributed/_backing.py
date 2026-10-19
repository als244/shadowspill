"""Explicit sharding ownership, independent of parameter/gradient replication."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch.distributed as dist
from torch import nn

from ._control import Control
from ._ownership import Parameter, members


@dataclass(frozen=True)
class TransferSharding:
    """Borrow the groups defining one sharded CPU representation.

    ``group`` gathers a complete execution tensor from its CPU shards.
    ``replica_group`` joins owners of the same shard in different shard groups;
    it is not the full data-parallel group. With a single shard group it is None.
    The parameter's existing replica/contribution declarations retain their
    meaning. This layout affects computation weights and optionally gradients;
    it does not change optimizer-state or master-parameter ownership. With
    ``shard_optimizer=True``, those remain sharded over parameter replicas.
    All groups are caller-owned and must outlive the Runtime.
    """

    group: dist.ProcessGroup
    replica_group: dist.ProcessGroup | None = None
    host_group: dist.ProcessGroup | None = None

    def record(self) -> dict[str, Any]:
        return {
            "shard_group": list(members(self.group)),
            "shard_replicas": list(members(self.replica_group)),
        }


def resolve_backing(
    model: nn.Module,
    parameters: Sequence[Parameter],
    default: TransferSharding | None,
    overrides: Iterable[tuple[Iterable[nn.Parameter | str], TransferSharding | None]],
) -> dict[str, TransferSharding]:
    named = dict(model.named_parameters(remove_duplicate=False))
    canonical = {id(value): name for name, value in reversed(tuple(named.items()))}
    selected = {name: default for name in canonical.values()}
    supplied: dict[str, dict[str, Any] | None] = {}
    for names, layout in overrides:
        if layout is not None and not isinstance(layout, TransferSharding):
            raise TypeError("sharding override requires TransferSharding or None")
        if isinstance(names, (str, nn.Parameter)):
            raise TypeError(
                "a sharding override selects an iterable of names/parameters"
            )
        for name in names:
            parameter = named.get(name) if isinstance(name, str) else name
            if id(parameter) not in canonical:
                raise ValueError("sharding override selects an unregistered parameter")
            key = canonical[id(parameter)]
            record = None if layout is None else layout.record()
            if key in supplied and supplied[key] != record:
                raise ValueError(f"conflicting sharding overrides for {key}")
            supplied[key] = record
            selected[key] = layout
    resolved = {name: layout for name, layout in selected.items() if layout is not None}
    for declaration in parameters:
        layout = resolved.get(declaration.name)
        if layout is None:
            continue
        shard, replicas = members(layout.group), members(layout.replica_group)
        if str(dist.get_backend(layout.group)).lower() != "nccl":
            raise ValueError("transfer sharding requires an NCCL execution group")
        if layout.host_group is not None and (
            str(dist.get_backend(layout.host_group)).lower() != "gloo"
            or members(layout.host_group) != shard
        ):
            raise ValueError("sharding host_group must be Gloo with the shard members")
        if dist.get_rank() not in shard or dist.get_rank() not in replicas:
            raise ValueError(f"{declaration.name}: rank is outside its sharding groups")
        if not set(shard) <= set(declaration.replicas) or not set(replicas) <= set(
            declaration.replicas
        ):
            raise ValueError(
                f"{declaration.name}: sharding groups exceed parameter replicas"
            )
    return resolved


def validate_backing(
    control: Control,
    parameters: Sequence[Parameter],
    layouts: Mapping[str, TransferSharding],
) -> None:
    """Check shard partitions and matching owners using all participants' records."""
    records = control.exchange(
        "sharding/layouts",
        {
            p.name: {
                "replicas": list(p.replicas),
                "layout": layouts[p.name].record() if p.name in layouts else None,
            }
            for p in parameters
        },
    )
    peers = dict(zip(control.members, records, strict=True))
    for values in peers.values():
        for name, value in values.items():
            layout = value["layout"]
            if layout is None:
                continue
            replicas = value["replicas"]
            partitions = set()
            owner_groups = {}
            for peer in replicas:
                other = peers[peer][name]["layout"]
                if other is None:
                    raise ValueError(
                        f"{name}: replicas disagree about transfer sharding"
                    )
                shard = tuple(other["shard_group"])
                if peer not in shard or not set(shard) <= set(replicas):
                    raise ValueError(f"{name}: invalid shard partition")
                partitions.add(shard)
                owner_groups[peer] = tuple(other["shard_replicas"])
                for member in shard:
                    record = peers[member][name]["layout"]
                    if record is None or tuple(record["shard_group"]) != shard:
                        raise ValueError(f"{name}: shard members disagree")
            sizes = {len(group) for group in partitions}
            if len(sizes) != 1 or sum(map(len, partitions)) != len(replicas):
                raise ValueError(
                    f"{name}: shard groups must partition replicas equally"
                )
            ordered = sorted(partitions)
            for group in ordered:
                for position, member in enumerate(group):
                    expected = {other[position] for other in ordered}
                    if set(owner_groups[member]) != expected:
                        raise ValueError(
                            f"{name}: matching shard owners are inconsistent"
                        )
