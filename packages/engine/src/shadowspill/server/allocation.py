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
    """The compute a `Server` was given, as a snapshot of its environment.

    It describes resources already assigned, one local machine or a
    single-node Slurm step, and never requests any. Construction probes no
    device: the service discovers the node's devices and memory when it
    starts, and refuses a multi-node Slurm allocation, which needs a node
    daemon on every node.

    Parameters
    ----------
    environment : mapping of str to str
        Environment variables that describe the assignment, such as
        ``CUDA_VISIBLE_DEVICES`` and Slurm's. Copied without
        ``SHADOWSPILL_TOKEN``; the service's preparation and worker
        processes inherit it. Left out of ``repr``.
    pool : str, default "default"
        Compute pool of the node's devices, as hardware listings show it and
        `Resources` selects it. Not empty, at most 256 characters.
    controller_only : bool, default False
        Whether this host is only the controller, holding the service's
        processing and storage and offering none of its devices.

    Raises
    ------
    TypeError
        If ``controller_only`` is not a bool, or ``environment`` holds a name
        or value that is not a str.
    ValueError
        If ``pool`` is empty, has surrounding spaces or is longer than 256
        characters.
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
        """Describe the resources assigned to this process's environment.

        Parameters
        ----------
        pool : str, default "default"
            Compute pool of the node's devices.

        Returns
        -------
        Allocation
            An allocation of ``os.environ``.
        """
        return cls(os.environ, pool)

    @classmethod
    def controller(cls) -> Allocation:
        """Describe a host that is only the controller.

        The service runs its processing and storage here and offers none of
        this host's devices. Registered node daemons take no runs, so no run
        can start.

        Returns
        -------
        Allocation
            A controller-only allocation of ``os.environ``, in pool
            ``"default"``.
        """
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
