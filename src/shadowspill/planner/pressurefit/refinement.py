"""The capacity ladder PressureFit climbs when admission does not fit.

Admission can refuse a schedule that the simulator accepted, because the
simulator reasons about capacity and admission reasons about addresses. When
that happens the answer is to hand PressureFit less object capacity and try
again, and this decides how much less.
"""

from __future__ import annotations

from dataclasses import replace

from shadowspill.simulator import SimulationConfig

from ..admission import AdmissionFacts

_ADMISSION_RESERVE_GRANULARITY_BYTES = 2 << 20
_ADMISSION_INITIAL_REFINEMENT_BYTES = 128 << 20
_ADMISSION_DOUBLING_LIMIT_BYTES = 1 << 30
_ADMISSION_LINEAR_REFINEMENT_BYTES = 512 << 20


def scheduled_admission_refinement(attempt: int) -> int:
    """Return the deterministic reserve increment for one failed admission."""

    doubled = _ADMISSION_INITIAL_REFINEMENT_BYTES << attempt
    if doubled <= _ADMISSION_DOUBLING_LIMIT_BYTES:
        return doubled
    attempts_after_limit = attempt - 3
    return (
        _ADMISSION_DOUBLING_LIMIT_BYTES
        + attempts_after_limit * _ADMISSION_LINEAR_REFINEMENT_BYTES
    )


def round_up_admission_reserve(value: int) -> int:
    granularity = _ADMISSION_RESERVE_GRANULARITY_BYTES
    return ((value + granularity - 1) // granularity) * granularity


def with_object_capacity(
    config: SimulationConfig,
    admission: AdmissionFacts,
    capacity_bytes: int,
    *,
    shared_execution_bytes: int,
) -> tuple[SimulationConfig, AdmissionFacts]:
    devices = tuple(
        replace(
            device,
            capacity_bytes=capacity_bytes + shared_execution_bytes,
        )
        if device.device_id == admission.device_id
        else device
        for device in config.devices
    )
    if devices == config.devices:
        raise ValueError(
            f"admission device {admission.device_id!r} is absent from simulation"
        )
    return (
        replace(config, devices=devices),
        replace(admission, object_capacity_bytes=capacity_bytes),
    )


__all__ = [
    "round_up_admission_reserve",
    "scheduled_admission_refinement",
    "with_object_capacity",
]
