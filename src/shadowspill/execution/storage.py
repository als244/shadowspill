"""Metadata-only CUDA views and bounded access to persistent spill storage."""

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, cast

import torch

from shadowspill.artifacts.plan import Plan
from shadowspill.ir import MemoryLocation
from shadowspill.runtime.plan import RuntimeBridge


@dataclass(frozen=True)
class TensorChunk:
    """A contiguous region of physical tensor storage, in dtype elements.

    ``shape`` describes the original tensor. ``offset`` indexes its storage,
    not a flattened logical view; tied/strided views share one initialization.
    """

    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    offset: int
    storage_elements: int


def dtype_of(geometry: Mapping[str, Any]) -> torch.dtype:
    return cast(torch.dtype, getattr(torch, geometry["dtype"].removeprefix("torch.")))


class Storage:
    def __init__(self, plan: Plan, bridge: RuntimeBridge, device: torch.device):
        self.plan, self.bridge, self.device = plan, bridge, device
        self.aliases = {o.object_id: o.alias_group_id for o in plan.program.objects}
        self.sizes = {a.alias_group_id: a.size_bytes for a in plan.program.alias_groups}
        self.owners: dict[str, torch.Tensor] = {}
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
        self.initial = {r.alias_group_id for r in plan.schedule.initial_residency}
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
        for alias in self.initial:
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

    def state_entries(self) -> Iterator[tuple[str, str, Mapping[str, Any] | None]]:
        bindings = self.plan.program.bindings
        for item in bindings["registrations"]:
            name = item["name"]
            if item.get("component_path"):
                name += "." + ".".join(item["component_path"])
            yield name, item["object_id"], None
        for item in bindings.get("optimizer_bindings", ()):
            if (
                item["role"] not in {"parameter", "gradient"}
                and item["object_id"] is not None
            ):
                yield item["name"], item["object_id"], item.get("initialization")
        if bindings["kind"] == "forward_backward":
            for item in bindings["gradients"]:
                yield (
                    "gradient." + item["parameter_name"],
                    item["gradient_object_id"],
                    None,
                )

    def visit_state(
        self,
        callback: Callable[[TensorChunk, torch.Tensor], None],
        chunk_bytes: int,
        *,
        initialize: bool,
    ) -> None:
        if chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be positive")
        visited = set()
        for name, object_id, initializer in self.state_entries():
            alias = self.aliases[object_id]
            if initialize and alias not in self.initial:
                continue
            if alias in visited:
                continue
            visited.add(alias)
            geometry = self.geometry[object_id]
            dtype = dtype_of(geometry)
            owner = self.host(alias).view(dtype)
            width = max(1, chunk_bytes // dtype.itemsize)
            for start in range(0, owner.numel(), width):
                destination = owner[start : start + width]
                request = TensorChunk(
                    name, tuple(geometry["shape"]), dtype, start, owner.numel()
                )
                if initialize and initializer is not None:
                    if initializer["kind"] not in {"constant", "value"}:
                        raise NotImplementedError(f"unsupported initializer for {name}")
                    destination.fill_(initializer["value"])
                else:
                    callback(request, destination)

    def clear_views(self) -> None:
        self.outputs.clear()
        self.owners.clear()
