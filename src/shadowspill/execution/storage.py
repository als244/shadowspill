"""Metadata-only CUDA views and bounded access to persistent spill storage."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, cast

import torch

from shadowspill.artifacts.plan import Plan
from shadowspill.artifacts.tensors import tensor_geometry
from shadowspill.ir import MemoryLocation
from shadowspill.parallelism.views import owned_view_offsets
from shadowspill.runtime.plan.bridge import RuntimeBridge

from .host_values import HostValues
from .spill import SpillStorage
from .tensor_regions import tensor_regions
from .views import StorageViews


def dtype_of(geometry: Mapping[str, Any]) -> torch.dtype:
    return cast(torch.dtype, getattr(torch, geometry["dtype"].removeprefix("torch.")))


class Storage:
    def __init__(
        self,
        plan: Plan,
        bridge: RuntimeBridge,
        device: torch.device,
        *,
        shared: Storage | None = None,
        shared_aliases: Mapping[str, str] | None = None,
        extra_initial: Iterable[str] = (),
        logical_rank: int | None = None,
        transfer_descriptors: Mapping[str, int] | None = None,
    ):
        self.plan, self.bridge, self.device = plan, bridge, device
        self.spill = SpillStorage(
            bridge,
            {
                a.alias_group_id: a.size_bytes
                if a.spill_size_bytes is None
                else a.spill_size_bytes
                for a in plan._costed_program.alias_groups
            },
        )
        if logical_rank is None:
            if (
                plan.program.parallelism is not None
                and plan.program.parallelism.world_size > 1
            ):
                raise ValueError(
                    "distributed storage requires an explicit logical_rank"
                )
            logical_rank = 0
        self.owned_offsets = owned_view_offsets(plan.program, logical_rank)
        self.regions = tensor_regions(plan.program, logical_rank)
        self.aliases = {o.object_id: o.alias_group_id for o in plan.program.objects}
        self.sizes = {a.alias_group_id: a.size_bytes for a in plan.program.alias_groups}
        self.shared_aliases = dict(shared_aliases or {})
        if shared is not None and shared_aliases is None:
            self.shared_aliases = {alias: alias for alias in self.sizes}
        self.owners: StorageViews = StorageViews(
            None if shared is None else shared.owners, self.shared_aliases
        )
        if shared is not None:
            registered = shared.bridge.objects.registered_runtime_objects()
            bridge.objects.borrow(
                {
                    alias: registered[source]
                    for alias, source in self.shared_aliases.items()
                    if source in registered
                },
            )
        self.outputs: dict[str, torch.Tensor] = {}
        self.host_values = HostValues(plan)
        self.geometry = tensor_geometry(plan.program)
        self.initial = {
            r.alias_group_id for r in plan.schedule.initial_residency
        } | set(extra_initial)
        if any(
            r.location is not MemoryLocation.SPILL
            for r in plan.schedule.initial_residency
        ):
            raise NotImplementedError(
                "saved Executor currently requires initial spill residency"
            )
        retain = {
            a.alias_group_id: a.retain_spill_copy for a in plan.program.alias_groups
        }
        for alias in self.initial - bridge.objects.registered_aliases():
            bridge.objects.reserve_spill(
                alias,
                retain_spill_copy=retain[alias],
                transfer_address=(transfer_descriptors or {}).get(alias, 0),
            )
        for alias, descriptor in (transfer_descriptors or {}).items():
            # Gradients can be first produced by backward. Register their
            # transfer behavior without inventing an initial spill value.
            bridge.objects.register_placeholder(alias, transfer_address=descriptor)

    def owner(self, alias: str) -> torch.Tensor:
        if alias not in self.owners:
            self.owners[alias] = torch.ops.shadowspill._make_unbacked_cuda_storage(
                torch.empty(0, device=self.device), self.sizes[alias]
            )
        return self.owners[alias]

    def argument(self, object_id: str, geometry: Mapping[str, Any]) -> torch.Tensor:
        if geometry["device_type"] == "cpu":
            return self.host_values.get(object_id)
        return torch.empty(0, dtype=dtype_of(geometry), device=self.device).set_(
            self.owner(self.aliases[object_id]).untyped_storage(),
            geometry.get("storage_offset", 0) + self.owned_offsets.get(object_id, 0),
            geometry["shape"],
            geometry["stride"],
        )

    def clear_views(self) -> None:
        self.outputs.clear()
        self.host_values.values.clear()
        self.owners.local.clear()
