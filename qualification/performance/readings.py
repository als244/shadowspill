"""Portable artifact identities and runtime counter differences."""

import hashlib
from pathlib import Path
from typing import Any


def _runtime_delta(before: Any, after: Any) -> dict[str, int]:
    return {
        "device_allocations": int(
            after.backend.device_allocations - before.backend.device_allocations
        ),
        "pinned_host_registrations": int(
            after.backend.pinned_host_registrations
            - before.backend.pinned_host_registrations
        ),
        "event_driver_creates": int(
            after.runtime.event_lease_driver_creates
            - before.runtime.event_lease_driver_creates
        ),
        "event_growth_rejections": int(
            after.runtime.event_lease_growth_rejections
            - before.runtime.event_lease_growth_rejections
        ),
        "allocation_callbacks": int(
            after.allocation_callbacks - before.allocation_callbacks
        ),
        "free_callbacks": int(after.free_callbacks - before.free_callbacks),
        "fetch_transfers": int(
            after.runtime.fetch_transfers - before.runtime.fetch_transfers
        ),
        "evict_transfers": int(
            after.runtime.evict_transfers - before.runtime.evict_transfers
        ),
        "bytes_fetched": int(
            after.runtime.bytes_fetched - before.runtime.bytes_fetched
        ),
        "bytes_evicted": int(
            after.runtime.bytes_evicted - before.runtime.bytes_evicted
        ),
    }


def _artifact_identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
