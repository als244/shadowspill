"""Existing allocation settings, independent of model setup or client connections."""

from __future__ import annotations

import os
import socket
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from shadowspill.orchestration.launch.admission import NodeAdmission, NodeCapacity
from shadowspill.orchestration.launch.host_memory import host_memory
from shadowspill.orchestration.launch.node_capacity import discover_capacity
from shadowspill_client._records import text


@dataclass(frozen=True)
class Allocation:
    """A snapshot of the service's assigned environment, never a scheduler request.

    The current deployment adapter serves one local node or assigned single-node
    Slurm step. Multi-node bootstrap must supply real node daemons and is not inferred
    from a controller's visible devices. Construction performs no device probe.
    """

    environment: Mapping[str, str] = field(repr=False)
    pool: str = "default"
    controller_only: bool = False

    def __post_init__(self) -> None:
        if type(self.controller_only) is not bool:
            raise TypeError("controller_only must be boolean")
        if len(text(self.pool, "compute pool")) > 256:
            raise ValueError("compute pool exceeds its size bound")
        if not isinstance(self.environment, Mapping) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in self.environment.items()
        ):
            raise TypeError("allocation environment requires string names and values")
        environment = dict(self.environment)
        environment.pop("SHADOWSPILL_TOKEN", None)
        object.__setattr__(self, "environment", MappingProxyType(environment))

    @classmethod
    def from_environment(cls, *, pool: str = "default") -> Allocation:
        return cls(os.environ, pool)

    @classmethod
    def controller(cls) -> Allocation:
        """Use this host only for service CPU/storage; node daemons supply all GPUs."""
        return cls(os.environ, controller_only=True)

    def _local_node(self, root: Path, *, controller_host_bytes: int) -> NodeAdmission:
        if self.controller_only:
            memory = host_memory()
            available = min(memory.capacity_bytes, memory.available_bytes)
            if controller_host_bytes > available:
                raise ValueError("controller cannot fit its fixed host reservation")
            return NodeAdmission(
                NodeCapacity(
                    socket.gethostname(),
                    (),
                    available - controller_host_bytes,
                    len(os.sched_getaffinity(0)),
                )
            )
        # Refuse an incomplete deployment before probing or starting workers.
        for key in ("SLURM_JOB_NUM_NODES", "SLURM_NNODES"):
            value = self.environment.get(key)
            if value is not None and (not value.isdecimal() or int(value) != 1):
                raise NotImplementedError(
                    "multi-node service requires a node daemon on every node; "
                    "the local adapter cannot use only the controller node"
                )
        return NodeAdmission(
            discover_capacity(
                environment=self.environment,
                root=root,
                reserved_host_bytes=controller_host_bytes,
            )
        )
