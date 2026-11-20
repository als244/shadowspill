"""Explicit per-worker host accounting for an atomic node reservation."""

from collections.abc import Mapping
from dataclasses import dataclass

from shadowspill.orchestration.launch.admission import ResourceRequest
from shadowspill_client._records import integer
from shadowspill_client.protocol.codec import MAX_METADATA_BYTES

from .protocol import WorkerStartup


def startup_metadata_bytes(workers: int) -> int:
    """Retained launch descriptors, in addition to streaming transfer scratch."""
    integer(workers, "startup workers", minimum=1)
    return 4096 + 8 * MAX_METADATA_BYTES * workers


@dataclass(frozen=True)
class WorkerHost:
    """Host memory allowance of each training worker, beyond its pools and staging.

    Neither value is inferred from idle memory use: a deployment qualifies
    both for its installed components. A worker's trace memory is counted
    separately, from its `WorkerStartup`; node daemon and controller memory
    and retained files are separate reservations.

    Parameters
    ----------
    process_bytes : int
        Resident process overhead, in bytes: the interpreter, the framework
        and compiled runtime, thread stacks and imported components.
    artifact_bytes : int
        Peak, in bytes, of loading, decoding and rechecking the saved Plan
        bundle, with checkpoint descriptors, manifest encoding and default
        random-state snapshots. Not the compressed file size.

    Raises
    ------
    ValueError
        If either is not an int of at least 1.
    """

    process_bytes: int
    artifact_bytes: int

    def __post_init__(self) -> None:
        integer(self.process_bytes, "worker process allowance", minimum=1)
        integer(self.artifact_bytes, "decoded artifact allowance", minimum=1)

    def peak_bytes(self, startup: WorkerStartup) -> int:
        """Return a worker's host memory peak, which the service reserves for it.

        Parameters
        ----------
        startup : WorkerStartup
            The worker's startup record.

        Returns
        -------
        int
            In bytes: its spill pool, staging and trace memory, this
            allowance, and fixed command and log buffers.

        Raises
        ------
        TypeError
            If ``startup`` is not a `WorkerStartup`.
        """
        if not isinstance(startup, WorkerStartup):
            raise TypeError("worker accounting requires validated startup evidence")
        # Input, initialization and output copies execute serially and share the
        # admitted staging maximum. Command decoding can coexist with it. Log
        # drains belong to the parent, but still consume node memory per child.
        return (
            startup.spill_bytes
            + startup.staging_bytes
            + startup.diagnostics_bytes
            + self.process_bytes
            + self.artifact_bytes
            + 8 * MAX_METADATA_BYTES
            + 2 * (64 << 10)
        )


def worker_request(
    workers: Mapping[str, tuple[WorkerStartup, WorkerHost]],
) -> ResourceRequest:
    """Count every physical worker once, reserving the whole node group together."""
    if not isinstance(workers, Mapping) or not 0 < len(workers) <= 65536:
        raise ValueError("provide a bounded nonempty assigned worker set")
    devices, host, ranks = {}, 0, set()
    first: WorkerStartup | None = None
    for device, (startup, allowance) in workers.items():
        if not isinstance(startup, WorkerStartup) or not isinstance(
            allowance, WorkerHost
        ):
            raise TypeError("worker resources require startup and host allowances")
        if first is None:
            first = startup
        if (
            startup.scope.rank in ranks
            or startup.scope.run_id != first.scope.run_id
            or startup.scope.server_incarnation != first.scope.server_incarnation
            or startup.bundle != first.bundle
            or startup.logical_by_worker != first.logical_by_worker
        ):
            raise ValueError("assigned workers disagree on run, bundle or rank mapping")
        ranks.add(startup.scope.rank)
        devices[device] = startup.device_bytes
        host += allowance.peak_bytes(startup)
    return ResourceRequest(devices, host)
