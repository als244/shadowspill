from __future__ import annotations

import pytest

from shadowspill.ir import (
    AliasGroupSpec,
    DeviceSpec,
    MemoryLocation,
    MemorySchedule,
    ObjectSpec,
    Program,
    ResidencySpec,
    ResourceKind,
    ResourceSpec,
    SharedResidencyPolicy,
    TaskProfile,
    TaskSpec,
    shared_residency_footprint,
)
from shadowspill.simulator import (
    SimulationConfig,
    SimulationInfeasibleError,
    simulate,
)


def _shared_input_program(*, output_bytes: int) -> Program:
    return Program(
        devices=(DeviceSpec("device_0", "process_0", "device", 0),),
        alias_groups=(
            AliasGroupSpec(
                "shared_storage",
                "device_0",
                64,
                retain_spill_copy=True,
                shared_residency=SharedResidencyPolicy.SHARED_READ_ONLY,
            ),
            AliasGroupSpec("output_storage", "device_0", output_bytes),
        ),
        objects=(
            ObjectSpec("shared", "shared_storage", 0, 64),
            ObjectSpec("output", "output_storage", 0, output_bytes),
        ),
        profiles=(TaskProfile("task_profile", 10, 0, "task_contract"),),
        tasks=(
            TaskSpec(
                "task",
                ResourceSpec("device_0", ResourceKind.COMPUTE),
                "task_profile",
                inputs=("shared",),
                outputs=("output",),
            ),
        ),
    )


def _config() -> SimulationConfig:
    return SimulationConfig.single_device(
        "device_0",
        device_capacity_bytes=96,
        spill_capacity_bytes=64,
        fetch_bandwidth_bytes_per_second=1_000_000,
        evict_bandwidth_bytes_per_second=1_000_000,
    )


def test_shared_footprint_is_charged_once_outside_movable_aliases() -> None:
    program = _shared_input_program(output_bytes=32)
    footprint = shared_residency_footprint(program)
    schedule = MemorySchedule(
        initial_residency=(),
        actions=(),
        final_residency=(ResidencySpec("output_storage", MemoryLocation.DEVICE),),
    )

    result = simulate(program, schedule, config=_config())

    assert footprint.device_bytes == (("device_0", 64),)
    assert footprint.spill_bytes == 64
    assert footprint.alias_group_ids == ("shared_storage",)
    assert result.device_peak("device_0").object_bytes == 96
    assert result.device_peak("device_0").total_bytes == 96
    assert result.spill_peak_bytes == 64
    assert result.transfer_intervals == ()


def test_shared_footprint_reduces_capacity_available_to_task_outputs() -> None:
    program = _shared_input_program(output_bytes=33)
    schedule = MemorySchedule(
        initial_residency=(),
        actions=(),
        final_residency=(ResidencySpec("output_storage", MemoryLocation.DEVICE),),
    )

    with pytest.raises(SimulationInfeasibleError) as caught:
        simulate(program, schedule, config=_config())

    assert caught.value.kind == "task-device-capacity"


def test_shared_footprint_must_fit_physical_execution_and_spill_budgets() -> None:
    program = _shared_input_program(output_bytes=0)
    too_small_device = SimulationConfig.single_device(
        "device_0",
        device_capacity_bytes=63,
        spill_capacity_bytes=64,
        fetch_bandwidth_bytes_per_second=1,
        evict_bandwidth_bytes_per_second=1,
    )
    too_small_spill = SimulationConfig.single_device(
        "device_0",
        device_capacity_bytes=64,
        spill_capacity_bytes=63,
        fetch_bandwidth_bytes_per_second=1,
        evict_bandwidth_bytes_per_second=1,
    )
    schedule = MemorySchedule((), ())

    with pytest.raises(ValueError, match="shared residency requires"):
        simulate(program, schedule, config=too_small_device)
    with pytest.raises(ValueError, match="shared spill residency exceeds"):
        simulate(program, schedule, config=too_small_spill)
