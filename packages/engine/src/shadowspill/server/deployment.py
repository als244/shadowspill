"""Deployment limits for existing service owners; these are not model settings."""

from dataclasses import dataclass, field

from shadowspill.server.startup.configuration import LocalCreationLimits
from shadowspill.training.worker.resources import WorkerHost
from shadowspill_client._records import integer, number

from .artifact_uploads import UPLOAD_TTL_SECONDS
from .http import HTTP_SCRATCH_BYTES


@dataclass(frozen=True)
class ServerLimits:
    """Explicit process/storage envelopes, with defaults for the qualified local path.

    Accounting reservations are not OS quotas or universal peak-memory proofs.
    Deployments qualify process bounds for their installed definitions. Training
    requests still choose their own per-worker device/spill and input budgets.
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
    temporary_disk_bytes: int = 32 << 30
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
            max_preparation_files=1024,
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
        # Each artifact owner has its fixed index plus bounded transfer scratch.
        return (16 << 20) * (
            1 + self.max_artifact_owners
        ) + self.request_host_bytes * self.max_open_records

    @property
    def fixed_host_bytes(self) -> int:
        return (
            self.storage_bytes
            + self.job_host_bytes * (self.max_training_runs + 1)
            + HTTP_SCRATCH_BYTES * self.max_http_requests
        )
