"""Bind node observations to explicit local or existing scheduler assignment."""

import os
import re
import socket
from collections.abc import Mapping
from pathlib import Path

from shadowspill_client._records import integer
from shadowspill_client.protocol.requests import identifier

from .admission import AssignedDevice, NodeCapacity, ResourcesUnavailable
from .device_inventory import MAX_DEVICES, DeviceInventory, probe_devices
from .host_memory import HostMemory, host_memory
from .slurm_job import SlurmJob, query_job


def _decimal(environment: Mapping[str, str], name: str, *, minimum: int = 0) -> int:
    value = environment.get(name, "")
    if re.fullmatch(r"[0-9]{1,20}", value) is None or int(value) < minimum:
        raise ValueError(f"assigned node requires an explicit numeric {name}")
    return int(value)


def _step(environment: Mapping[str, str], node_id: str) -> tuple[int, int] | None:
    if not any(
        name in environment
        for name in (
            "SLURM_JOB_ID",
            "SLURM_JOBID",
            "SLURM_STEP_ID",
            "SLURMD_NODENAME",
            "SLURM_GPUS_ON_NODE",
        )
    ):
        return None
    job_id = _decimal(environment, "SLURM_JOB_ID", minimum=1)
    if "SLURM_JOBID" in environment and _decimal(environment, "SLURM_JOBID") != job_id:
        raise ValueError("Slurm job environment identities disagree")
    # A batch script or allocation shell is not a per-node GPU assignment. Node daemons
    # run in explicit numeric steps; their local CUDA mask is supplied by Slurm.
    _decimal(environment, "SLURM_STEP_ID")
    count = _decimal(environment, "SLURM_GPUS_ON_NODE")
    if count > MAX_DEVICES or environment.get("SLURMD_NODENAME") != node_id:
        raise ValueError("Slurm node identity or assigned GPU count is invalid")
    if count and not environment.get("CUDA_VISIBLE_DEVICES", "").strip():
        raise ValueError("assigned Slurm GPU step has no explicit CUDA visibility")
    return job_id, count


def assigned_capacity(
    node_id: str,
    devices: DeviceInventory,
    memory: HostMemory,
    *,
    reserved_host_bytes: int,
    environment: Mapping[str, str],
    job: SlurmJob | None = None,
) -> NodeCapacity:
    """Make a ledger ceiling from trusted observations in this node's context.

    Host claims fit physical/cgroup capacity, the initial availability snapshot
    and any scheduler ceiling, less explicit controller/node-daemon fixed allowances.
    Artifact arenas and worker claims are added later through NodeAdmission.
    Device totals and free snapshots are different facts: this function does not
    claim that an observed GPU is idle or establish ownership over other jobs.
    The allocation adapter owns exclusive slots; CUDA-visible UUIDs identify them.
    """
    identifier(node_id, "assigned node ID")
    integer(reserved_host_bytes, "fixed node host reservation")
    if not isinstance(devices, DeviceInventory) or not isinstance(memory, HostMemory):
        raise TypeError(
            "node capacity requires validated device and memory observations"
        )
    step = _step(environment, node_id)
    ceiling = memory.capacity_bytes
    cpus = len(os.sched_getaffinity(0))
    if step is None:
        if job is not None:
            raise ValueError("Slurm allocation facts require an assigned local step")
    else:
        if (
            job is None
            or job.job_id != step[0]
            or job.user_id != os.getuid()
            or node_id not in job.nodes
        ):
            raise ValueError("assigned Slurm step lacks matching owner/job/node facts")
        if len(devices.devices) != step[1]:
            raise ValueError("CUDA-visible devices differ from the assigned step count")
        ceiling = min(ceiling, job.nodes[node_id].memory_bytes)
        cpus = min(cpus, job.nodes[node_id].cpus)
    if reserved_host_bytes > ceiling:
        raise ValueError("fixed node host reservation cannot fit the allocation")
    available = min(ceiling, memory.available_bytes)
    if reserved_host_bytes > available:
        raise ResourcesUnavailable(
            "fixed node host reservation exceeds current availability"
        )
    return NodeCapacity(
        node_id,
        tuple(
            AssignedDevice(device.uuid, device.uuid, device.total_bytes, device.name)
            for device in devices.devices
        ),
        available - reserved_host_bytes,
        cpus,
        devices.driver,
    )


def discover_capacity(
    *,
    environment: Mapping[str, str],
    root: Path,
    reserved_host_bytes: int,
    node_id: str | None = None,
    timeout: float = 30,
) -> NodeCapacity:
    """Discover only this node daemon's node; a controller cannot invent remote slots.

    The existing Slurm allocation launcher must start one node daemon inside an assigned
    step per node. This read-only helper never starts a scheduler job or step.
    Local deployment explicitly supplies its inherited CUDA visibility, if any.
    """
    if node_id is None:
        node_id = environment.get("SLURMD_NODENAME", socket.gethostname())
    identifier(node_id, "assigned node ID")
    integer(reserved_host_bytes, "fixed node host reservation")
    step = _step(environment, node_id)
    job = (
        None
        if step is None
        else query_job(step[0], environment=environment, root=root, timeout=timeout)
    )
    memory = host_memory()
    devices = probe_devices(environment=environment, root=root, timeout=timeout)
    return assigned_capacity(
        node_id,
        devices,
        memory,
        reserved_host_bytes=reserved_host_bytes,
        environment=environment,
        job=job,
    )
