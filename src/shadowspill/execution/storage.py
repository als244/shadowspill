"""Metadata-only CUDA views and bounded access to persistent spill storage."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, cast

import torch

from shadowspill.artifacts.plan import Plan
from shadowspill.ir import MemoryLocation
from shadowspill.runtime.plan import RuntimeBridge


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
        extra_initial: Iterable[str] = (),
    ):
        self.plan, self.bridge, self.device = plan, bridge, device
        self.aliases = {o.object_id: o.alias_group_id for o in plan.program.objects}
        self.sizes = {a.alias_group_id: a.size_bytes for a in plan.program.alias_groups}
        self.owners: dict[str, torch.Tensor] = {} if shared is None else shared.owners
        if shared is not None:
            bridge.objects.adopt_registered(
                shared.bridge.objects.registered_runtime_objects(), borrowed=True
            )
        self.outputs: dict[str, torch.Tensor] = {}
        self.geometry: dict[str, Mapping[str, Any]] = {}
        tasks = {t.task_id: t for t in plan.program.tasks}
        for entry in plan.program.bindings["entrypoints"]:
            description = plan.program.bindings["task_inputs"][
                tasks[entry["task_id"]].profile_id
            ]
            geometry = dict(
                zip(
                    description["tensor_positions"],
                    description["tensor_geometry"],
                    strict=True,
                )
            )
            for slot in entry["input_slots"]:
                self.geometry.setdefault(
                    slot["object_id"], geometry[slot["leaf_index"]]
                )
        for item in plan.program.bindings.get("optimizer_bindings", ()):
            if item["object_id"] is not None:
                self.geometry.setdefault(
                    item["object_id"], {**item, "storage_offset": 0}
                )
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
            bridge.objects.reserve_spill(alias, retain_spill_copy=retain[alias])

    def owner(self, alias: str) -> torch.Tensor:
        if alias not in self.owners:
            self.owners[alias] = torch.ops.shadowspill._make_unbacked_cuda_storage(
                torch.empty(0, device=self.device), self.sizes[alias]
            )
        return self.owners[alias]

    def argument(self, object_id: str, geometry: Mapping[str, Any]) -> torch.Tensor:
        return torch.empty(0, dtype=dtype_of(geometry), device=self.device).set_(
            self.owner(self.aliases[object_id]).untyped_storage(),
            geometry.get("storage_offset", 0),
            geometry["shape"],
            geometry["stride"],
        )

    def host(self, alias: str) -> torch.Tensor:
        location = self.bridge.objects.spill_location(alias)
        if location is None:
            raise RuntimeError(f"no addressable spill lease for {alias}")
        address, size = location
        return cast(
            torch.Tensor,
            torch.ops.shadowspill._make_runtime_cpu_storage(
                torch.empty(0, dtype=torch.uint8),
                self.bridge.spill_pool_id,
                address,
                self.bridge.objects.runtime_object_id(alias),
                size,
            ),
        )

    def clear_views(self) -> None:
        self.outputs.clear()
        self.owners.clear()
