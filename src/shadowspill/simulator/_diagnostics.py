"""Shared diagnostic formatting for simulator implementations."""

from __future__ import annotations

from shadowspill._status import Status

_STATUS_KIND: dict[int, str] = {
    Status.INVALID_ARGUMENT: "invalid-argument",
    Status.INTERNAL_FAILURE: "allocation-failure",
    Status.INITIAL_DEVICE_CAPACITY: "initial-device-capacity",
    Status.INITIAL_HOST_CAPACITY: "initial-host-capacity",
    Status.TASK_INPUT_DEADLOCK: "task-input-deadlock",
    Status.TASK_DEVICE_CAPACITY: "task-device-capacity",
    Status.PREFETCH_DEVICE_CAPACITY: "prefetch-device-capacity",
    Status.OFFLOAD_HOST_CAPACITY: "offload-host-capacity",
    Status.TRANSFER_DEADLOCK: "transfer-deadlock",
    Status.INVALID_RELEASE: "invalid-release",
    Status.RELEASE_TRANSFER_CONFLICT: "release-transfer-conflict",
    Status.INVALID_OFFLOAD: "invalid-offload",
    Status.INVALID_PREFETCH: "invalid-prefetch",
    Status.FINAL_RESIDENCY: "final-residency",
    Status.SIMULATION_INTERNAL_ERROR: "internal-error",
}

_STATUS_MESSAGE: dict[int, str] = {
    Status.INVALID_ARGUMENT: "invalid argument",
    Status.INTERNAL_FAILURE: "allocation failure",
    Status.INITIAL_DEVICE_CAPACITY: "initial device capacity exceeded",
    Status.INITIAL_HOST_CAPACITY: "initial host capacity exceeded",
    Status.TASK_INPUT_DEADLOCK: "task input cannot become resident",
    Status.TASK_DEVICE_CAPACITY: (
        "task output and workspace exceed device capacity"
    ),
    Status.PREFETCH_DEVICE_CAPACITY: "prefetch cannot reserve device capacity",
    Status.OFFLOAD_HOST_CAPACITY: "offload cannot reserve host capacity",
    Status.TRANSFER_DEADLOCK: "transfer has no progress source",
    Status.INVALID_RELEASE: "release has no ready device copy",
    Status.RELEASE_TRANSFER_CONFLICT: "release conflicts with a transfer",
    Status.INVALID_OFFLOAD: "offload has no ready device source",
    Status.INVALID_PREFETCH: (
        "prefetch has no host source or duplicates a device copy"
    ),
    Status.FINAL_RESIDENCY: "required final residency was not reached",
    Status.SIMULATION_INTERNAL_ERROR: "internal simulator invariant failed",
}

_CAPACITY_STATUSES: frozenset[int] = frozenset(
    (
        Status.INITIAL_DEVICE_CAPACITY,
        Status.INITIAL_HOST_CAPACITY,
        Status.TASK_DEVICE_CAPACITY,
        Status.PREFETCH_DEVICE_CAPACITY,
        Status.OFFLOAD_HOST_CAPACITY,
    )
)


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
