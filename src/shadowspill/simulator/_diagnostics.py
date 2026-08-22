"""Shared diagnostic formatting for simulator implementations."""

from __future__ import annotations

_STATUS_KIND = {
    1: "invalid-argument",
    2: "allocation-failure",
    3: "initial-device-capacity",
    4: "initial-host-capacity",
    5: "task-input-deadlock",
    6: "task-device-capacity",
    7: "prefetch-device-capacity",
    8: "offload-host-capacity",
    9: "transfer-deadlock",
    10: "invalid-release",
    11: "release-transfer-conflict",
    12: "invalid-offload",
    13: "invalid-prefetch",
    14: "final-residency",
    15: "internal-error",
}

_STATUS_MESSAGE = {
    1: "invalid argument",
    2: "allocation failure",
    3: "initial device capacity exceeded",
    4: "initial host capacity exceeded",
    5: "task input cannot become resident",
    6: "task output and workspace exceed device capacity",
    7: "prefetch cannot reserve device capacity",
    8: "offload cannot reserve host capacity",
    9: "transfer has no progress source",
    10: "release has no ready device copy",
    11: "release conflicts with a transfer",
    12: "offload has no ready device source",
    13: "prefetch has no host source or duplicates a device copy",
    14: "required final residency was not reached",
    15: "internal simulator invariant failed",
}

_CAPACITY_STATUSES = frozenset((3, 4, 6, 7, 8))


def simulation_status_kind(status: int) -> str:
    """Return the stable machine-readable kind for one simulator status."""

    return _STATUS_KIND.get(status, "unknown")


def simulation_failure_detail(
    status: int,
    *,
    time_ns: int,
    error_device: int,
    error_location: int,
    capacity_bytes: int,
    used_bytes: int,
    requested_bytes: int,
    device_ids: tuple[str, ...],
) -> str:
    """Format the same failure detail for Python and compiled simulation."""

    if status not in _CAPACITY_STATUSES:
        return _STATUS_MESSAGE.get(status, f"simulator status {status}")
    location = (
        "host"
        if error_location == 1
        else f"device:{device_ids[error_device]}"
    )
    return (
        f"{simulation_status_kind(status)} at {time_ns} ns: "
        f"{used_bytes} used + {requested_bytes} requested exceeds "
        f"{capacity_bytes} bytes at {location}"
    )


__all__ = ["simulation_failure_detail", "simulation_status_kind"]
