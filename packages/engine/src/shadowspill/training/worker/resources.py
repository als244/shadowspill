"""Explicit per-worker host accounting for an atomic node reservation."""

from collections.abc import Mapping
from dataclasses import dataclass

from shadowspill.orchestration.launch.admission import ResourceRequest
from shadowspill_client._records import integer
from shadowspill_client.protocol.codec import MAX_METADATA_BYTES

from .protocol import WorkerStartup


@dataclass(frozen=True)
class WorkerHost:
    """Deployment allowances in addition to exact admitted pools and staging.

    ``process_bytes`` covers Python, framework/compiled runtime, thread stacks,
    imported definitions and other resident process overhead. ``artifact_bytes``
    covers the peak of loading, decoding and rechecking the saved bundle, plus
    checkpoint descriptors/manifest encoding and bounded default RNG snapshots
    while that family remains resident;
    it is not the compressed file size. Neither value is inferred from idle RSS.
    The deployment must qualify these allowances for its installed definitions.
    Trace rings, owned copies and report scratch are included separately in the
    startup's diagnostics allowance. Agent/controller arenas and retained files
    are separately owned reservations.
    """

    process_bytes: int
    artifact_bytes: int

    def __post_init__(self) -> None:
        integer(self.process_bytes, "worker process allowance", minimum=1)
        integer(self.artifact_bytes, "decoded artifact allowance", minimum=1)

    def peak_bytes(self, startup: WorkerStartup) -> int:
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
