"""Isolated regression for trigger-time prefetch destination reservation."""

from __future__ import annotations

import pytest

from shadowspill.ir import (
    AliasGroupSpec,
    DeviceSpec,
    MemoryAction,
    MemoryActionKind,
    MemoryLocation,
    MemorySchedule,
    ObjectRole,
    ObjectSpec,
    Program,
    ResidencySpec,
    ResourceKind,
    ResourceSpec,
    TaskProfile,
    TaskSpec,
)
from shadowspill.simulator import (
    SimulationConfig,
    SimulationInfeasibleError,
    TransferDirection,
    simulate,
)


def _program() -> Program:
    compute = ResourceSpec("cuda_0", ResourceKind.COMPUTE)
    return Program(
        devices=(DeviceSpec("cuda_0", "process_0", "cuda", 0),),
        alias_groups=(
            AliasGroupSpec("resident_model_state", "cuda_0", 20),
            AliasGroupSpec("earlier_prefetch", "cuda_0", 40),
            AliasGroupSpec("future_optimizer_state", "cuda_0", 40),
            AliasGroupSpec("head_gradient", "cuda_0", 30),
        ),
        objects=(
            ObjectSpec(
                "resident_parameter",
                "resident_model_state",
                0,
                20,
                ObjectRole.PARAMETER,
            ),
            ObjectSpec(
                "earlier_state",
                "earlier_prefetch",
                0,
                40,
                ObjectRole.OPTIMIZER_STATE,
            ),
            ObjectSpec(
                "future_state",
                "future_optimizer_state",
                0,
                40,
                ObjectRole.OPTIMIZER_STATE,
            ),
            ObjectSpec(
                "gradient_output",
                "head_gradient",
                0,
                30,
                ObjectRole.GRADIENT,
            ),
        ),
        profiles=(
            TaskProfile("trigger_profile", 10, 0, "trigger_abi"),
            TaskProfile("head_profile", 10, 20, "head_abi"),
            TaskProfile("optimizer_profile", 10, 0, "optimizer_abi"),
        ),
        tasks=(
            TaskSpec(
                "earlier_transfer_trigger",
                compute,
                "trigger_profile",
                inputs=("resident_parameter",),
            ),
            TaskSpec(
                "future_optimizer_prefetch_trigger",
                compute,
                "trigger_profile",
                dependencies=("earlier_transfer_trigger",),
                inputs=("resident_parameter",),
            ),
            TaskSpec(
                "current_head_computation",
                compute,
                "head_profile",
                dependencies=("future_optimizer_prefetch_trigger",),
                inputs=("resident_parameter",),
                outputs=("gradient_output",),
            ),
            TaskSpec(
                "future_optimizer_update",
                compute,
                "optimizer_profile",
                dependencies=("current_head_computation",),
                inputs=("future_state",),
            ),
        ),
    )


def _schedule() -> MemorySchedule:
    return MemorySchedule(
        initial_residency=(
            ResidencySpec("resident_model_state", MemoryLocation.DEVICE),
            ResidencySpec("earlier_prefetch", MemoryLocation.HOST),
            ResidencySpec("future_optimizer_state", MemoryLocation.HOST),
        ),
        actions=(
            MemoryAction(
                "earlier_transfer_trigger",
                "earlier_prefetch",
                MemoryActionKind.PREFETCH,
            ),
            MemoryAction(
                "future_optimizer_prefetch_trigger",
                "future_optimizer_state",
                MemoryActionKind.PREFETCH,
            ),
            MemoryAction(
                "current_head_computation",
                "head_gradient",
                MemoryActionKind.RELEASE,
            ),
        ),
        final_residency=(
            ResidencySpec("future_optimizer_state", MemoryLocation.DEVICE),
        ),
    )


def _config(capacity: int) -> SimulationConfig:
    return SimulationConfig.single_device(
        "cuda_0",
        device_capacity_bytes=capacity,
        host_capacity_bytes=80,
        fetch_bandwidth_bytes_per_second=1_000_000_000,
        evict_bandwidth_bytes_per_second=1_000_000_000,
    )


@pytest.mark.parametrize("record_timeline", [False, True])
def test_transfer_start_charging_would_admit_but_causal_admission_rejects(
    record_timeline: bool,
) -> None:
    # Under the old rule, the future 40-byte destination was not charged while
    # it waited behind the earlier FETCH. The current head task therefore appeared
    # to fit: 20 resident + 40 active-FETCH + 30 output + 20 workspace = 110 <= 120.
    old_apparent_head_demand = 20 + 40 + 30 + 20
    assert old_apparent_head_demand <= 120

    # At the later predicted FETCH start, the head output would already have been
    # released, so the old model also thought the future copy fit:
    # 20 resident + 40 earlier state + 40 future state = 100 <= 120.
    old_apparent_transfer_demand = 20 + 40 + 40
    assert old_apparent_transfer_demand <= 120

    # Causal admission reserves both FETCH destinations when their triggers
    # complete. The current task actually needs 20 + 40 + 40 + 30 + 20 = 150.
    with pytest.raises(SimulationInfeasibleError) as caught:
        simulate(
            _program(),
            _schedule(),
            config=_config(120),
            record_timeline=record_timeline,
        )

    error = caught.value
    assert error.kind == "task-device-capacity"
    assert error.task_id == "current_head_computation"
    assert error.used_bytes == 100
    assert error.requested_bytes == 50
    assert error.capacity_bytes == 120


def test_causal_reservation_preserves_overlap_when_the_plan_really_fits() -> None:
    result = simulate(
        _program(),
        _schedule(),
        config=_config(150),
        record_timeline=True,
    )

    head = next(
        item
        for item in result.task_intervals
        if item.task_id == "current_head_computation"
    )
    first_fetch, future_fetch = result.transfer_intervals
    assert first_fetch.direction is TransferDirection.FETCH
    assert future_fetch.direction is TransferDirection.FETCH
    assert (head.start_ns, head.end_ns) == (20, 30)
    assert first_fetch.start_ns == 10
    assert first_fetch.end_ns == 50
    assert head.start_ns < first_fetch.end_ns
    assert (future_fetch.start_ns, future_fetch.end_ns) == (50, 90)
    assert result.device_peak("cuda_0").total_bytes == 150
