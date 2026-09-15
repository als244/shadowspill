"""Measure the runtime's transfer routes, and read the matrix it publishes.

The runtime owns the measurement and the published matrix; this module is the
two neutral calls that drive them and the decoding of what comes back. The
guards -- an open runtime, no callable or in-progress plan -- are the
runtime's, and stay on it.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
from collections.abc import Sequence
from typing import Any

from shadowspill.status import ABI_VERSION

from .abi import (
    TransferCalibrationConfig,
    TransferRouteKey,
    runtime_library,
)
from .abi import (
    TransferProfile as RuntimeTransferProfile,
)
from .configuration import RuntimeConfigurationError
from .topology import TransferCapabilities, TransferProfile

#: What `TransferProfile.provenance` records about when a route was measured.
INITIALIZATION_PROVENANCE = 0
RECALIBRATION_PROVENANCE = 1


def calibrate(
    handle: int,
    pool_names: Sequence[str],
    *,
    routes: Sequence[tuple[str, str]] | None,
    provenance: int,
    small_copy_bytes: int = 4096,
    large_copy_bytes: int = 256 << 20,
    warmup_copies: int = 4,
    measured_copies: int = 16,
) -> None:
    """Measure all or selected routes; the runtime publishes the new matrix."""

    keys: Any = None
    count = 0
    if routes is not None:
        encoded: list[TransferRouteKey] = []
        for source, destination in routes:
            try:
                source_id = pool_names.index(source)
                destination_id = pool_names.index(destination)
            except ValueError as exc:
                raise RuntimeConfigurationError(
                    f"unknown transfer route {(source, destination)!r}"
                ) from exc
            encoded.append(TransferRouteKey(source_id, destination_id))
        count = len(encoded)
        keys = (TransferRouteKey * count)(*encoded) if count else None
    config = TransferCalibrationConfig(
        abi_version=ABI_VERSION,
        small_copy_bytes=small_copy_bytes,
        large_copy_bytes=large_copy_bytes,
        warmup_copies=warmup_copies,
        measured_copies=measured_copies,
        provenance=provenance,
    )
    status = int(
        runtime_library().shadowspill_runtime_calibrate_transfer_capabilities(
            handle, ctypes.byref(config), keys, count
        )
    )
    if status != 0:
        raise RuntimeConfigurationError(
            f"transfer calibration failed with status {status}"
        )


def read_transfer_capabilities(
    handle: int, pool_names: Sequence[str]
) -> TransferCapabilities:
    """The matrix the runtime currently publishes, as one immutable snapshot."""

    count = ctypes.c_uint32()
    generation = ctypes.c_uint64()
    status = int(
        runtime_library().shadowspill_runtime_transfer_profiles(
            handle,
            None,
            0,
            ctypes.byref(count),
            ctypes.byref(generation),
        )
    )
    if status != 0:
        raise RuntimeConfigurationError(
            f"transfer-profile size query failed with status {status}"
        )
    records = (RuntimeTransferProfile * count.value)()
    status = int(
        runtime_library().shadowspill_runtime_transfer_profiles(
            handle,
            records,
            count.value,
            ctypes.byref(count),
            ctypes.byref(generation),
        )
    )
    if status != 0:
        raise RuntimeConfigurationError(
            f"transfer-profile read failed with status {status}"
        )
    profiles = tuple(
        TransferProfile(
            source=pool_names[item.source_pool_id],
            destination=pool_names[item.destination_pool_id],
            source_pool_id=int(item.source_pool_id),
            destination_pool_id=int(item.destination_pool_id),
            generation=int(item.generation),
            latency_nanoseconds=int(item.latency_nanoseconds),
            bandwidth_bytes_per_second=int(item.bandwidth_bytes_per_second),
            solo_bandwidth_bytes_per_second=int(item.solo_bandwidth_bytes_per_second),
            concurrent_bandwidth_bytes_per_second=int(
                item.concurrent_bandwidth_bytes_per_second
            ),
            solo_measurement_nanoseconds=int(item.solo_measurement_nanoseconds),
            concurrent_measurement_nanoseconds=int(
                item.concurrent_measurement_nanoseconds
            ),
            calibrated_timestamp_nanoseconds=int(item.calibrated_timestamp_nanoseconds),
            small_copy_bytes=int(item.small_copy_bytes),
            large_copy_bytes=int(item.large_copy_bytes),
            measured_copies=int(item.measured_copies),
            available=bool(item.available),
            calibrated=bool(item.calibrated),
            provenance=(
                "initialization"
                if int(item.provenance) == INITIALIZATION_PROVENANCE
                else "recalibration"
            ),
            calibration_mode={
                0: "identity",
                1: "solo",
                2: "bidirectional_concurrent",
            }.get(int(item.calibration_mode), "unknown"),
            concurrent_route_count=int(item.concurrent_route_count),
        )
        for item in records
    )
    canonical = {
        "generation": int(generation.value),
        "pool_names": list(pool_names),
        "profiles": [profile.as_dict() for profile in profiles],
    }
    digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return TransferCapabilities(
        generation=int(generation.value),
        pool_names=tuple(pool_names),
        profiles=profiles,
        digest=digest,
    )


__all__ = [
    "INITIALIZATION_PROVENANCE",
    "RECALIBRATION_PROVENANCE",
    "calibrate",
    "read_transfer_capabilities",
]
