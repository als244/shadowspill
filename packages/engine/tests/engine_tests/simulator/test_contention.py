"""Analytic overlap and shared-communicator timelines, through the public IR."""

from dataclasses import replace

import pytest

from shadowspill.ir import (
    AliasGroupSpec,
    DeviceSpec,
    MemoryAction,
    MemoryActionKind,
    MemoryLocation,
    MemorySchedule,
    ObjectSpec,
    ResidencySpec,
    ResourceKind,
    ResourceSpec,
    ShadowSpillProgram,
    TaskOverlapCost,
    TaskProfile,
    TaskResourcePhase,
    TaskSpec,
    TransferPhase,
    TransferSpec,
    ValidationError,
)
from shadowspill.simulator import simulate
from shadowspill.simulator.indexing.timing import interpolate_cost
from shadowspill.simulator.model import DeviceSimulationConfig, SimulationConfig


def case(
    task_ns=100,
    collective_ns=100,
    *,
    task_rate=0.5,
    collective_rate=0.5,
    copy_ns=0,
    compute_after=0,
    task_phases=(),
    enabled=True,
):
    costs = (
        (TaskOverlapCost("gather", 100, task_rate, collective_rate),) if enabled else ()
    )
    program = ShadowSpillProgram(
        (DeviceSpec("gpu", "rank", "cuda", 0),),
        (
            AliasGroupSpec(
                "weight",
                "gpu",
                100,
                transfer=TransferSpec(
                    100,
                    (
                        TransferPhase("copy", bytes=copy_ns),
                        TransferPhase(
                            "timed",
                            bytes=100,
                            duration_ns=collective_ns,
                            resource="nccl",
                            submission_ordered=True,
                            timing_class="gather",
                        ),
                    ),
                    (TransferPhase("copy", bytes=100),),
                ),
            ),
        ),
        (ObjectSpec("weight", "weight", 0, 100),),
        (
            TaskProfile("zero", 0, 0, "control"),
            TaskProfile("compute", task_ns, 0, "test", task_phases, costs),
            TaskProfile("use", compute_after, 0, "test", overlap_costs=costs),
        ),
        (
            TaskSpec("entry", ResourceSpec("gpu", ResourceKind.CONTROL), "zero"),
            TaskSpec(
                "compute",
                ResourceSpec("gpu", ResourceKind.COMPUTE),
                "compute",
                dependencies=("entry",),
            ),
            TaskSpec(
                "use",
                ResourceSpec("gpu", ResourceKind.COMPUTE),
                "use",
                dependencies=("compute",),
                inputs=("weight",),
            ),
        ),
    )
    schedule = MemorySchedule(
        (ResidencySpec("weight", MemoryLocation.SPILL),),
        (MemoryAction("entry", "weight", MemoryActionKind.FETCH),),
        (ResidencySpec("weight", MemoryLocation.DEVICE),),
    )
    config = SimulationConfig(
        (DeviceSimulationConfig("gpu", 100, 10**9, 10**9, 10**9, 10**9),), 100
    )
    return program, schedule, config


def replay(*args, **kwargs):
    program, schedule, config = case(*args, **kwargs)
    return simulate(program, schedule, config=config)


@pytest.mark.parametrize(
    "task,collective,kwargs,expected",
    [
        (100, 100, {}, 200),
        (100, 200, {}, 300),
        (200, 100, {}, 300),
        (200, 100, {"copy_ns": 100}, 300),
        (100, 100, {"copy_ns": 200}, 300),
        (100, 100, {"compute_after": 100}, 300),
        (100, 100, {"task_rate": 0.5, "collective_rate": 1}, 150),
        (100, 100, {"copy_ns": 50, "task_rate": 0.5, "collective_rate": 0.25}, 225),
        (100, 100, {"enabled": False}, 100),
        (100, 100, {"task_rate": 1, "collective_rate": 1}, 100),
    ],
)
def test_progress_changes_only_during_overlap(task, collective, kwargs, expected):
    assert replay(task, collective, **kwargs).makespan_ns == expected


@pytest.mark.parametrize(
    "copy,rate,expected", [(0, 1, 170), (50, 1, 220), (0, 0.5, 200)]
)
def test_task_reserves_communicator_only_during_its_own_phase(copy, rate, expected):
    phases = (
        TaskResourcePhase(30),
        TaskResourcePhase(40, "nccl"),
        TaskResourcePhase(30),
    )
    result = replay(
        copy_ns=copy, task_rate=rate, collective_rate=rate, task_phases=phases
    )
    assert result.makespan_ns == expected
    assert "transfer-resource" in result.task_intervals[1].stall_reasons


def test_task_phases_work_without_overlap_measurements():
    phases = (
        TaskResourcePhase(30),
        TaskResourcePhase(40, "nccl"),
        TaskResourcePhase(30),
    )
    assert replay(task_phases=phases, enabled=False).makespan_ns == 170


def test_compute_only_phases_do_not_reset_the_others_progress():
    phases = tuple(TaskResourcePhase(ns) for ns in (30, 40, 30))
    assert replay(task_phases=phases).makespan_ns == 200


def test_distinct_communicators_can_overlap():
    phases = (TaskResourcePhase(100, "other-nccl"),)
    assert replay(task_phases=phases).makespan_ns == 100


def test_measurements_round_trip_and_change_program_identity():
    program, _, _ = case()
    assert ShadowSpillProgram.from_json(program.to_json()) == program
    without = replace(
        program, profiles=tuple(replace(p, overlap_costs=()) for p in program.profiles)
    )
    assert program.digest != without.digest


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf"), 1.1, True])
def test_invalid_progress_rates_are_refused(rate):
    with pytest.raises(ValidationError):
        TaskOverlapCost("gather", 100, rate, 1)


def test_phase_duration_must_match_task():
    with pytest.raises(ValidationError):
        TaskProfile("p", 100, 0, "abi", (TaskResourcePhase(101),))


def test_log_size_interpolation_and_endpoint_clamping():
    samples = (
        TaskOverlapCost("gather", 0, 0.2, 0.4),
        TaskOverlapCost("gather", 99, 0.8, 1),
    )
    assert interpolate_cost(samples, "gather", 9) == pytest.approx((0.5, 0.7))
    assert interpolate_cost(samples, "gather", 10000) == (0.8, 1)
    assert interpolate_cost(samples, "scatter", 100) == (1, 1)
