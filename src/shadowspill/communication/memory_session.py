"""Retained communicator memory, excluding caller/probe tensors and native cache."""

from collections.abc import Mapping
from typing import Any

import torch

from shadowspill.profiling.device_memory import external_device_bytes

from .binding.groups import GroupBindings
from .requests import CollectiveRequest


class CommunicationMemory:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        torch.cuda.synchronize(device)
        self.native_before = torch.cuda.memory_allocated(device)
        self.external_before = external_device_bytes(device)

    def finish(
        self,
        bindings: GroupBindings | None,
        requests: tuple[CollectiveRequest, ...],
        setup: Mapping[str, Any],
        *,
        backend_setup: Mapping[str, Any] | None = None,
    ) -> tuple[int | None, dict[str, Any]]:
        torch.cuda.synchronize(self.device)
        external_after = external_device_bytes(self.device)
        native = max(0, torch.cuda.memory_allocated(self.device) - self.native_before)
        external = (
            None
            if self.external_before is None or external_after is None
            else max(0, external_after - self.external_before)
        )
        local: dict[str, Any] = {
            "logical_rank": 0 if bindings is None else bindings.logical_rank,
            "native_retained_bytes": native,
            "external_retained_bytes": external,
        }
        rows = [local] if bindings is None else bindings.gather(local)
        complete = all(row["external_retained_bytes"] is not None for row in rows)
        size = (
            max(
                row["native_retained_bytes"] + row["external_retained_bytes"]
                for row in rows
            )
            if complete
            else None
        )
        return size, {
            "memory_kind": "measured" if complete else "unresolved",
            "memory_scope": {
                "setup": dict(setup),
                "groups": sorted({request.ranks for request in requests}),
                "topology": {"node0": (0,)} if bindings is None else bindings.topology,
                "backend": "nccl",
                "backend_setup": dict(backend_setup or {}),
            },
            "memory_per_rank": sorted(rows, key=lambda row: row["logical_rank"]),
            "memory_source": "native live bytes + NVML external retained delta",
            "external_transient_kind": "unresolved",
            "memory_coverage": (
                "retained communicator resources and native task allocations; "
                "external transient scratch is not observed"
            ),
        }
