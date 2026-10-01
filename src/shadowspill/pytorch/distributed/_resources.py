"""Shared-host admission facts, checked before any spill pool is pinned.

The validation itself is independent of models, process-group backends or GPUs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Limit:
    # Physical memory uses a host identity; cgroup ancestors add their path.
    domain: str
    available_bytes: int


@dataclass(frozen=True)
class Resources:
    host: str
    device_id: str | None
    spill_bytes: int
    staging_bytes: int
    limits: tuple[Limit, ...]

    def record(self) -> dict[str, Any]:
        return asdict(self)


def validate_resources(records: Sequence[Resources]) -> None:
    devices: dict[tuple[str, str], int] = {}
    reservations: dict[tuple[str, str], int] = {}
    limits: dict[tuple[str, str], int] = {}
    for rank, record in enumerate(records):
        if record.spill_bytes < 0 or record.staging_bytes < 0:
            raise ValueError(f"rank {rank}: memory reservations must be nonnegative")
        if not record.host or not record.limits:
            raise ValueError(
                f"rank {rank}: host identity and memory limits are required"
            )
        if record.device_id is not None:
            identity = (record.host, record.device_id)
            if identity in devices:
                raise ValueError(
                    f"ranks {devices[identity]} and {rank} select the same "
                    f"physical device: {identity}"
                )
            devices[identity] = rank
        seen = set()
        for limit in record.limits:
            identity = (record.host, limit.domain)
            if limit.available_bytes < 0 or identity in seen:
                raise ValueError(f"rank {rank}: invalid or duplicate host-memory limit")
            seen.add(identity)
            reservations[identity] = (
                reservations.get(identity, 0)
                + record.spill_bytes
                + record.staging_bytes
            )
            limits[identity] = min(
                limits.get(identity, limit.available_bytes), limit.available_bytes
            )
    for identity, requested in reservations.items():
        if requested > limits[identity]:
            raise ValueError(
                f"combined spill/staging reservation exceeds host-memory "
                f"domain {identity}: "
                f"requested={requested}, available={limits[identity]}"
            )
