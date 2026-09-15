"""Small readings taken off the runtime and the plan report."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from shadowspill.pytorch import (
    Runtime,
)
from shadowspill.pytorch.runtime_adapter.bridge import (
    wait_idle,
)


def _phase_seconds(report: Any) -> dict[str, float]:
    return {
        name: int(nanoseconds) / 1e9 for name, nanoseconds in report.phase_timings_ns
    }


def _profile_metadata(microbatches: tuple[tuple[object, ...], ...]) -> list[object]:
    result: list[object] = []
    for microbatch in microbatches:
        lengths = microbatch[2]
        if not isinstance(lengths, Sequence):
            raise TypeError("performance sequence lengths must be a sequence")
        result.append({"sequence_lengths": list(lengths)})
    return result


def _wait_idle(training: Any) -> None:
    """Drain terminal actions at a qualification measurement boundary."""

    wait_idle(training._executor._bridge)


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


def _report_runtime_transfer_capabilities(runtime: Runtime) -> dict[str, object]:
    """Print and return the exact transfer measurements consumed by planning."""

    capabilities = runtime.transfer_capabilities
    print(
        "runtime transfer capabilities: "
        f"generation={capabilities.generation} digest={capabilities.digest}",
        flush=True,
    )
    for route_name, route in runtime.routes.items():
        profile = capabilities.route(route.source, route.destination)
        print(
            f"  {route_name} ({route.source}->{route.destination}): "
            f"effective={profile.bandwidth_bytes_per_second / 1e9:.3f} GB/s "
            f"concurrent={profile.concurrent_bandwidth_bytes_per_second / 1e9:.3f} "
            f"GB/s solo={profile.solo_bandwidth_bytes_per_second / 1e9:.3f} GB/s "
            f"latency={profile.latency_nanoseconds / 1e3:.3f} us "
            f"mode={profile.calibration_mode} "
            f"probe={profile.measured_copies}x"
            f"{profile.large_copy_bytes / (1 << 20):.0f} MiB",
            flush=True,
        )
    return capabilities.as_dict()


def _calibration_suspect(runtime: Runtime) -> bool:
    """Detect the degraded bidirectional-concurrent calibration mode."""

    capabilities = runtime.transfer_capabilities
    for route in runtime.routes.values():
        profile = capabilities.route(route.source, route.destination)
        solo = profile.solo_bandwidth_bytes_per_second
        concurrent = profile.concurrent_bandwidth_bytes_per_second
        if solo > 0 and concurrent < 0.65 * solo:
            return True
    return False
