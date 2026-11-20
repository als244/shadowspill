"""Deployment limits for existing service owners; these are not model settings."""

from dataclasses import dataclass, field

from shadowspill.server.startup.configuration import LocalCreationLimits
from shadowspill.training.worker.resources import WorkerHost
from shadowspill_client._records import integer, number

from .artifact_uploads import UPLOAD_TTL_SECONDS
from .http import HTTP_SCRATCH_BYTES


@dataclass(frozen=True)
class ServerLimits:
    """Concurrency, memory and storage envelopes of a `Server`.

    The defaults suit one local node. Each envelope is an accounting
    reservation, not an operating-system quota or a proof of peak use: a
    deployment qualifies its process bounds for the components it installs.
    Training requests still choose their own device, spill and input budgets.

    Parameters
    ----------
    controller_host_bytes : int, default 1 GiB
        Host memory, in bytes, held back from the node for the service
        process itself.
    max_training_runs : int, default 1
        Runs the service owns at once, preparation and planning jobs
        included. A further creation is refused until one ends.
    max_http_requests : int, default 4
        HTTP exchanges served at once; a further one is answered busy.
    max_nodes : int, default 32
        Node daemons registered at once, when the server has
        ``node_authentication``.
    job_host_bytes : int, default 512 MiB
        Host memory, in bytes, reserved for each owned run's creation and
        control, decoded Plans included. One more such envelope serves
        training previews.
    request_host_bytes : int, default 128 MiB
        Host memory, in bytes, of each open request journal, for its index
        and working memory.
    max_open_records : int, default 2
        Request journals open at once. Each live run keeps its own open and
        a lookup of another run opens one briefly, so it is at least
        ``max_training_runs``.
    max_artifact_owners : int, default 2
        Authenticated owners whose artifact stores are open at once.
    artifact_bytes : int, default 512 GiB
        Stored artifact bytes per owner, in bytes, pending uploads included.
    artifact_count : int, default 100000
        Stored artifacts per owner.
    upload_ttl_seconds : float, default 86400
        How long an incomplete upload is kept without a verified chunk, in
        seconds. Stored artifacts never expire.
    temporary_disk_bytes : int, default 128 GiB
        Disk space, in bytes, counted for the service's temporary files:
        preparation directories and workers' logs, outputs and traces. Nothing
        is preallocated.
    temporary_files : int, default 4096
        Temporary-file reservations held at once against that budget.
    request_timeout : float, default 120
        Longest HTTP exchange, in seconds. An artifact download gets this
        long for each block it sends, so a download that keeps moving
        finishes whatever its size.
    creation : LocalCreationLimits
        Envelopes of each run's creation and workers. The default allows
        8 GiB of preparation host memory, 16 GiB of preparation disk, a
        worker host allowance of 2 GiB for the process and 512 MiB for its
        artifacts, 64 MiB each of transfer, publication and log space, and
        65,536 preparation files.

    Raises
    ------
    ValueError
        If an integer envelope is not an int of at least 1,
        ``max_open_records`` is less than ``max_training_runs``, or
        ``request_timeout`` or ``upload_ttl_seconds`` is not a positive
        number.
    TypeError
        If ``creation`` is not a `LocalCreationLimits`.
    """

    controller_host_bytes: int = 1 << 30
    max_training_runs: int = 1
    max_http_requests: int = 4
    max_nodes: int = 32
    job_host_bytes: int = 512 << 20
    request_host_bytes: int = 128 << 20
    max_open_records: int = 2
    max_artifact_owners: int = 2
    artifact_bytes: int = 512 << 30
    artifact_count: int = 100000
    upload_ttl_seconds: float = UPLOAD_TTL_SECONDS
    # Preparation files coexist with all workers' output/trace reservations.
    # This is an accounting ceiling, not a preallocated disk file.
    temporary_disk_bytes: int = 128 << 30
    temporary_files: int = 4096
    # Initial metadata resolution can stream a large local checkpoint's hashes.
    # Match the HTTP service/client default rather than cutting it off at 30s.
    request_timeout: float = 120
    creation: LocalCreationLimits = field(
        default_factory=lambda: LocalCreationLimits(
            preparation_host_bytes=8 << 30,
            preparation_disk_bytes=16 << 30,
            worker=WorkerHost(2 << 30, 512 << 20),
            transfer_bytes=64 << 20,
            publication_bytes=64 << 20,
            log_bytes=64 << 20,
            # Workers whose programs differ keep one profiling-cache entry per
            # worker and task, so the count grows with both.
            max_preparation_files=1 << 16,
        )
    )

    def __post_init__(self) -> None:
        for name in (
            "controller_host_bytes",
            "max_training_runs",
            "max_http_requests",
            "max_nodes",
            "job_host_bytes",
            "request_host_bytes",
            "max_open_records",
            "max_artifact_owners",
            "artifact_bytes",
            "artifact_count",
            "temporary_disk_bytes",
            "temporary_files",
        ):
            integer(getattr(self, name), name, minimum=1)
        if self.max_open_records < self.max_training_runs:
            raise ValueError("open request owners must cover every live run")
        if not number(self.request_timeout, "request timeout", minimum=0):
            raise ValueError("request timeout must be positive")
        if not number(self.upload_ttl_seconds, "upload lifetime", minimum=0):
            raise ValueError("upload lifetime must be positive")
        if not isinstance(self.creation, LocalCreationLimits):
            raise TypeError("creation limits require LocalCreationLimits")

    @property
    def storage_bytes(self) -> int:
        """Host memory, in bytes, budgeted for the service's storage.

        16 MiB each for the run catalog and every open artifact store, with
        its index and transfer scratch, and ``request_host_bytes`` for each
        open request journal.
        """
        # Each artifact owner has its fixed index plus bounded transfer scratch.
        return (16 << 20) * (
            1 + self.max_artifact_owners
        ) + self.request_host_bytes * self.max_open_records

    @property
    def fixed_host_bytes(self) -> int:
        """Host memory, in bytes, that the service reserves when it starts.

        `storage_bytes`, one ``job_host_bytes`` envelope per run and one for
        previews, and just over 16 MiB of scratch per concurrent HTTP
        exchange. A server with ``node_authentication`` reserves connection
        scratch for its node daemons beyond this.
        """
        return (
            self.storage_bytes
            + self.job_host_bytes * (self.max_training_runs + 1)
            + HTTP_SCRATCH_BYTES * self.max_http_requests
        )
