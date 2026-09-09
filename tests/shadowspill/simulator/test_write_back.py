"""A write-back refreshes the spill copy and keeps the device copy."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import pytest

from reference.python.simulator import simulate_python
from shadowspill.ir import (
    MemoryAction,
    MemoryActionKind,
    MemorySchedule,
    ShadowSpillProgram,
)
from shadowspill.simulator import (
    SimulationConfig,
    SimulationInfeasibleError,
    SimulationResult,
    TransferDirection,
    simulate,
)
from tests.shadowspill.ir._examples import (
    release_behind_write_back_schedule,
    write_back_program,
    write_back_schedule,
)

from ._examples import calibrated_config

Simulate = Callable[
    [ShadowSpillProgram, MemorySchedule, SimulationConfig], SimulationResult
]


def _compiled(
    program: ShadowSpillProgram, schedule: MemorySchedule, config: SimulationConfig
) -> SimulationResult:
    return simulate(program, schedule, config=config)


def _reference(
    program: ShadowSpillProgram, schedule: MemorySchedule, config: SimulationConfig
) -> SimulationResult:
    return simulate_python(program, schedule, config=config)


SIMULATORS = pytest.mark.parametrize(
    "run", [_compiled, _reference], ids=["compiled", "reference"]
)


@SIMULATORS
def test_a_write_back_keeps_the_device_copy_and_refreshes_the_spill_copy(
    run: Simulate,
) -> None:
    result = run(write_back_program(), write_back_schedule(), calibrated_config())

    write_back, fetch = result.transfer_intervals
    assert write_back.kind is MemoryActionKind.WRITE_BACK
    assert write_back.direction is TransferDirection.EVICT
    assert (write_back.start_ns, write_back.end_ns) == (100, 228)
    assert fetch.kind is MemoryActionKind.FETCH
    assert (fetch.start_ns, fetch.end_ns) == (400, 528)
    # the copy took the evict lane and freed nothing
    assert result.device_peak("cuda_0").object_bytes == 128
    assert result.spill_peak_bytes == 128
    consume = result.task_intervals[-1]
    assert consume.task_id == "consume"
    assert consume.start_ns == 528
    assert result.makespan_ns == 628


@SIMULATORS
def test_a_release_waits_for_the_write_back_it_follows(run: Simulate) -> None:
    result = run(
        write_back_program(),
        release_behind_write_back_schedule(),
        calibrated_config(),
    )

    write_back, fetch = result.transfer_intervals
    assert (write_back.start_ns, write_back.end_ns) == (100, 228)
    # the release went through when the copy landed, and only then was the
    # fetch behind it submitted: it never waited on a source
    assert fetch.ready_ns == 228
    assert (fetch.start_ns, fetch.end_ns) == (228, 356)
    assert fetch.stall_reasons == ()
    assert result.device_peak("cuda_0").object_bytes == 128
    assert result.makespan_ns == 500


@SIMULATORS
def test_a_write_back_of_a_current_spill_copy_is_free(run: Simulate) -> None:
    schedule = replace(
        write_back_schedule(),
        actions=(
            MemoryAction("update", "state_storage", MemoryActionKind.WRITE_BACK),
            MemoryAction("spacer", "state_storage", MemoryActionKind.WRITE_BACK),
        ),
    )

    result = run(write_back_program(), schedule, calibrated_config())

    (write_back,) = result.transfer_intervals
    assert (write_back.start_ns, write_back.end_ns) == (100, 228)
    assert result.makespan_ns == 500


@SIMULATORS
def test_a_second_copy_of_an_object_in_flight_is_refused(run: Simulate) -> None:
    schedule = replace(
        write_back_schedule(),
        actions=(
            MemoryAction("update", "state_storage", MemoryActionKind.WRITE_BACK),
            MemoryAction("update", "state_storage", MemoryActionKind.WRITE_BACK),
        ),
    )

    with pytest.raises(SimulationInfeasibleError) as caught:
        run(write_back_program(), schedule, calibrated_config())

    assert caught.value.kind == "invalid-write-back"
    assert caught.value.alias_group_ids == ("state_storage",)
