"""Compose admitted node-owned children with the existing startup barrier."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from shadowspill.orchestration.launch.admission import NodeLease
from shadowspill.orchestration.launch.connected import ConnectedProcess
from shadowspill.orchestration.launch.processes import worker_command

from .channel import write_message
from .group import WorkerGroup
from .protocol import CODEC, WorkerLaunch
from .startup import WorkerStartupConnection, start_workers


@dataclass(frozen=True)
class WorkerProcess:
    """Owned by the supplied node lease; this record cannot release its resources."""

    child: ConnectedProcess
    startup: WorkerStartupConnection


def launch_worker(
    lease: NodeLease,
    device_id: str,
    launch: WorkerLaunch,
    *,
    environment: Mapping[str, str],
    files: Path,
    outputs: Path,
    cache: Path,
    directory: Path,
    cwd: Path,
    log_limit_bytes: int = 64 << 20,
    startup_timeout: float = 120,
) -> WorkerProcess:
    """Launch only inside an existing complete resource claim.

    The agent owns filesystem roots, definition packages, rendezvous and all file
    grants. Its claim includes process/artifact overhead beyond these pool/staging
    lower bounds. This helper never discovers or invents allocation capacity.
    """
    CODEC.encode(launch)  # Complete bounded metadata check before any child launch.
    startup = launch.startup
    if (
        lease.request.devices.get(device_id, 0) < startup.device_bytes
        or lease.request.host_bytes < startup.spill_bytes + startup.staging_bytes
    ):
        raise ValueError("worker pools/staging exceed the existing node claim")
    world = len(startup.logical_by_worker)
    if world > 1 and (
        not environment.get("MASTER_ADDR")
        or not environment.get("MASTER_PORT", "").isdecimal()
        or not 0 < int(environment["MASTER_PORT"]) < 65536
    ):
        raise ValueError("multiple workers require an explicit agent rendezvous")
    child = lease.launch(
        device_id,
        lambda descriptor: worker_command(
            "shadowspill.training.worker.bootstrap",
            (
                "--control-fd",
                descriptor,
                "--files",
                files,
                "--outputs",
                outputs,
                "--cache",
                cache,
            ),
        ),
        environment={
            **environment,
            "RANK": str(startup.logical_by_worker.index(startup.scope.rank)),
            "WORLD_SIZE": str(world),
            "LOCAL_RANK": "0",
            "LOCAL_WORLD_SIZE": "1",
        },
        directory=directory,
        cwd=cwd,
        log_limit_bytes=log_limit_bytes,
        startup_timeout=startup_timeout,
    )
    try:
        write_message(child.stream, launch)
        return WorkerProcess(
            child, WorkerStartupConnection(startup, child.stream, child.stream)
        )
    except BaseException:
        child.close()
        raise


def start_processes(
    processes: Mapping[int, WorkerProcess], *, stop: Callable[[], None]
) -> WorkerGroup:
    """Only verified all-worker readiness removes startup transport deadlines."""
    group = start_workers(
        {rank: process.startup for rank, process in processes.items()}, stop=stop
    )
    try:
        for process in processes.values():
            process.child.ready()
    except BaseException:
        group.close()
        raise
    return group
