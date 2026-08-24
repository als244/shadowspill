from __future__ import annotations

import importlib
from dataclasses import replace

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
    DeviceSimulationConfig,
    SimulationConfig,
    SimulationInfeasibleError,
    simulate,
)
from tests.shadowspill.ir._examples import (
    SAVE_SELECTION,
    representative_program,
    representative_schedule,
)

from ._examples import calibrated_config, overlap_program, overlap_schedule


def test_public_simulator_fails_closed_without_the_library(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    implementation = importlib.import_module("shadowspill.simulator")

    def missing_library() -> None:
        raise RuntimeError("the simulator unavailable")

    monkeypatch.setattr(implementation, "simulator_api", missing_library)
    with pytest.raises(RuntimeError, match="the simulator unavailable"):
        simulate(
            overlap_program(),
            overlap_schedule(),
            config=calibrated_config(),
        )


def test_initial_device_capacity_failure_is_structured() -> None:
    with pytest.raises(SimulationInfeasibleError) as caught:
        simulate(
            representative_program(),
            representative_schedule(),
            selections=SAVE_SELECTION,
            config=calibrated_config(device_capacity_bytes=319),
        )

    error = caught.value
    assert error.kind == "initial-device-capacity"
    assert error.location == "device:cuda_0"
    assert error.capacity_bytes == 319
    assert error.used_bytes == 320
    assert error.requested_bytes == 0


def test_initial_spill_capacity_failure_is_structured() -> None:
    program = Program(
        devices=(DeviceSpec("cuda_0", "process_0", "cuda", 0),),
        alias_groups=(AliasGroupSpec("spill_storage", "cuda_0", 128),),
        objects=(
            ObjectSpec("spill_object", "spill_storage", 0, 128, ObjectRole.INPUT),
        ),
        profiles=(),
        tasks=(),
    )
    schedule = MemorySchedule(
        initial_residency=(ResidencySpec("spill_storage", MemoryLocation.SPILL),),
        actions=(),
        final_residency=(ResidencySpec("spill_storage", MemoryLocation.SPILL),),
    )

    with pytest.raises(SimulationInfeasibleError) as caught:
        simulate(
            program,
            schedule,
            config=calibrated_config(spill_capacity_bytes=127),
        )

    assert caught.value.kind == "initial-spill-capacity"
    assert caught.value.used_bytes == 128


def test_pending_offload_reports_spill_capacity_root_cause() -> None:
    with pytest.raises(SimulationInfeasibleError) as caught:
        simulate(
            overlap_program(),
            overlap_schedule(),
            config=calibrated_config(
                device_capacity_bytes=512,
                spill_capacity_bytes=127,
            ),
        )

    error = caught.value
    assert error.kind == "offload-spill-capacity"
    assert error.alias_group_ids == ("activation_storage",)
    assert error.capacity_bytes == 127
    assert error.requested_bytes == 128


def test_pending_prefetch_reports_device_capacity_root_cause() -> None:
    compute = ResourceSpec("cuda_0", ResourceKind.COMPUTE)
    program = Program(
        devices=(DeviceSpec("cuda_0", "process_0", "cuda", 0),),
        alias_groups=(
            AliasGroupSpec("resident_storage", "cuda_0", 64),
            AliasGroupSpec("spill_storage", "cuda_0", 128),
        ),
        objects=(
            ObjectSpec("resident", "resident_storage", 0, 64, ObjectRole.INPUT),
            ObjectSpec("spill_object", "spill_storage", 0, 128, ObjectRole.INPUT),
        ),
        profiles=(TaskProfile("trigger_profile", 10, 0, "trigger_abi"),),
        tasks=(
            TaskSpec(
                "trigger",
                compute,
                "trigger_profile",
                inputs=("resident",),
            ),
        ),
    )
    schedule = MemorySchedule(
        initial_residency=(
            ResidencySpec("resident_storage", MemoryLocation.DEVICE),
            ResidencySpec("spill_storage", MemoryLocation.SPILL),
        ),
        actions=(MemoryAction("trigger", "spill_storage", MemoryActionKind.PREFETCH),),
        final_residency=(ResidencySpec("spill_storage", MemoryLocation.DEVICE),),
    )

    with pytest.raises(SimulationInfeasibleError) as caught:
        simulate(
            program,
            schedule,
            config=calibrated_config(device_capacity_bytes=128),
        )

    error = caught.value
    assert error.kind == "prefetch-device-capacity"
    assert error.used_bytes == 64
    assert error.requested_bytes == 128


def test_prefetch_reserves_capacity_at_trigger_before_lane_head() -> None:
    compute = ResourceSpec("cuda_0", ResourceKind.COMPUTE)
    program = Program(
        devices=(DeviceSpec("cuda_0", "process_0", "cuda", 0),),
        alias_groups=(
            AliasGroupSpec("resident_storage", "cuda_0", 64),
            AliasGroupSpec("first_spill_storage", "cuda_0", 64),
            AliasGroupSpec("second_spill_storage", "cuda_0", 64),
        ),
        objects=(
            ObjectSpec("resident", "resident_storage", 0, 64, ObjectRole.INPUT),
            ObjectSpec("first", "first_spill_storage", 0, 64, ObjectRole.INPUT),
            ObjectSpec("second", "second_spill_storage", 0, 64, ObjectRole.INPUT),
        ),
        profiles=(TaskProfile("trigger_profile", 10, 0, "trigger_abi"),),
        tasks=(
            TaskSpec(
                "first_trigger",
                compute,
                "trigger_profile",
                inputs=("resident",),
            ),
            TaskSpec(
                "second_trigger",
                compute,
                "trigger_profile",
                dependencies=("first_trigger",),
                inputs=("resident",),
            ),
        ),
    )
    schedule = MemorySchedule(
        initial_residency=(
            ResidencySpec("resident_storage", MemoryLocation.DEVICE),
            ResidencySpec("first_spill_storage", MemoryLocation.SPILL),
            ResidencySpec("second_spill_storage", MemoryLocation.SPILL),
        ),
        actions=(
            MemoryAction(
                "first_trigger", "first_spill_storage", MemoryActionKind.PREFETCH
            ),
            MemoryAction(
                "second_trigger", "second_spill_storage", MemoryActionKind.PREFETCH
            ),
        ),
    )

    with pytest.raises(SimulationInfeasibleError) as caught:
        simulate(
            program,
            schedule,
            config=calibrated_config(device_capacity_bytes=160),
        )

    error = caught.value
    assert error.kind == "prefetch-device-capacity"
    assert error.time_ns == 20
    assert error.task_id == "second_trigger"
    assert error.alias_group_ids == ("second_spill_storage",)
    assert error.used_bytes == 128
    assert error.requested_bytes == 64


def test_task_workspace_and_outputs_are_admitted_together() -> None:
    with pytest.raises(SimulationInfeasibleError) as caught:
        simulate(
            representative_program(),
            representative_schedule(),
            selections=SAVE_SELECTION,
            config=calibrated_config(device_capacity_bytes=527),
        )

    error = caught.value
    assert error.kind == "task-device-capacity"
    assert error.task_id == "consume"
    assert error.used_bytes == 448
    assert error.requested_bytes == 80
    assert error.capacity_bytes == 527


def test_configuration_must_cover_program_devices_exactly() -> None:
    config = SimulationConfig(
        devices=(
            DeviceSimulationConfig(
                "other_device",
                1024,
                1_000_000_000,
                1_000_000_000,
            ),
        ),
        spill_capacity_bytes=1024,
    )

    with pytest.raises(ValueError, match="exactly match"):
        simulate(
            representative_program(),
            representative_schedule(),
            selections=SAVE_SELECTION,
            config=config,
        )


@pytest.mark.parametrize(
    "config",
    [
        lambda: DeviceSimulationConfig("", 1, 1, 1),
        lambda: DeviceSimulationConfig("cuda_0", -1, 1, 1),
        lambda: DeviceSimulationConfig("cuda_0", 1, 0, 1),
        lambda: SimulationConfig((), 1),
        lambda: SimulationConfig(
            (
                DeviceSimulationConfig("cuda_0", 1, 1, 1),
                DeviceSimulationConfig("cuda_0", 1, 1, 1),
            ),
            1,
        ),
        lambda: SimulationConfig(
            (DeviceSimulationConfig("cuda_0", 1, 1, 1),),
            -1,
        ),
        lambda: SimulationConfig((object(),), 1),
    ],
)
def test_invalid_configuration_is_rejected(config: object) -> None:
    with pytest.raises(ValueError):
        config()


def test_omitted_terminal_release_leaves_device_copy_resident() -> None:
    schedule = representative_schedule()
    retained = replace(schedule, actions=schedule.actions[:-1])

    result = simulate(
        representative_program(),
        retained,
        selections=SAVE_SELECTION,
        config=calibrated_config(device_capacity_bytes=600),
    )

    assert result.makespan_ns == 278
    assert result.device_peak("cuda_0").object_bytes == 512


def test_unknown_device_peak_raises_key_error() -> None:
    result = simulate(
        representative_program(),
        representative_schedule(),
        selections=SAVE_SELECTION,
        config=calibrated_config(device_capacity_bytes=600),
    )

    with pytest.raises(KeyError):
        result.device_peak("missing")


def test_relaxed_capacity_matches_enforced_capacity_when_the_plan_fits() -> None:
    """A plan that fits never trips a capacity check, so relaxing changes nothing.

    This is what makes the relaxed makespan usable as a bound: it agrees
    exactly with the enforced makespan at the point a plan becomes
    admissible, so a search pruning on it prunes on the real number.
    """

    arguments = {
        "selections": SAVE_SELECTION,
        "config": calibrated_config(device_capacity_bytes=600),
    }
    enforced = simulate(
        representative_program(),
        representative_schedule(),
        **arguments,
    )
    relaxed = simulate(
        representative_program(),
        representative_schedule(),
        relax_capacity=True,
        **arguments,
    )

    assert relaxed.makespan_ns == enforced.makespan_ns
    assert relaxed.device_peak("cuda_0") == enforced.device_peak("cuda_0")
    assert [item.stall_reasons for item in relaxed.task_intervals] == [
        item.stall_reasons for item in enforced.task_intervals
    ]


def test_relaxed_capacity_times_a_plan_that_does_not_fit() -> None:
    """An overflowing plan still has a duration, and relaxed mode reports it.

    Enforced simulation abandons at the first violation, so it can say only
    that the plan does not fit. The same plan simulated without capacity
    enforcement runs to completion, and its peak records how far over it went.
    """

    arguments = {
        "selections": SAVE_SELECTION,
        "config": calibrated_config(device_capacity_bytes=319),
    }
    with pytest.raises(SimulationInfeasibleError):
        simulate(representative_program(), representative_schedule(), **arguments)

    relaxed = simulate(
        representative_program(),
        representative_schedule(),
        relax_capacity=True,
        **arguments,
    )

    assert relaxed.makespan_ns > 0
    assert relaxed.device_peak("cuda_0").object_bytes > 319


def test_relaxed_capacity_names_every_violation_with_a_reason() -> None:
    """Enforced simulation names one violation; relaxed names them all.

    Each carries where and when it happened and which question the plan
    got wrong, so a repair can see whether a plan overflows in one place
    or everywhere without re-simulating once per violation.
    """

    relaxed = simulate(
        representative_program(),
        representative_schedule(),
        selections=SAVE_SELECTION,
        config=calibrated_config(device_capacity_bytes=319),
        relax_capacity=True,
    )

    assert relaxed.capacity_violations
    # The count is the true total, so truncation is always visible.
    assert relaxed.capacity_violation_count == len(relaxed.capacity_violations)
    first = relaxed.capacity_violations[0]
    assert first.reason == "initial-device-capacity"
    assert first.location == "device"
    assert first.device_id == "cuda_0"
    assert first.capacity_bytes == 319
    assert first.used_bytes == 320
    assert first.excess_bytes == 1


def test_a_plan_that_fits_records_no_violations() -> None:
    relaxed = simulate(
        representative_program(),
        representative_schedule(),
        selections=SAVE_SELECTION,
        config=calibrated_config(device_capacity_bytes=600),
        relax_capacity=True,
    )

    assert relaxed.capacity_violations == ()
    assert relaxed.capacity_violation_count == 0
