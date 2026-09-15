"""A result read back from a store re-encodes into the simulator's index space."""

from __future__ import annotations

import json
from dataclasses import asdict

from shadowspill.planner.serialization import _simulation_result_from_value
from shadowspill.simulator.indexing import (
    IntervalArrays,
    index_simulation_template,
    interval_arrays_from_result,
    simulate_template,
)

from ._examples import calibrated_config, overlap_program, overlap_schedule


def _rows(arrays: IntervalArrays) -> tuple[set[tuple[int, ...]], set[tuple[int, ...]]]:
    tasks = {
        (
            item.task,
            item.ready_ns,
            item.start_ns,
            item.end_ns,
            item.workspace_bytes,
            item.stall_mask,
        )
        for item in arrays.task_intervals[: arrays.task_interval_count]
    }
    transfers = {
        (
            item.alias,
            item.trigger_task,
            item.device,
            item.direction,
            item.kind,
            item.sequence,
            item.ready_ns,
            item.start_ns,
            item.end_ns,
            item.bytes,
            item.stall_mask,
        )
        for item in arrays.transfer_intervals[: arrays.transfer_interval_count]
    }
    return tasks, transfers


def test_projected_intervals_match_the_simulators_own() -> None:
    program = overlap_program()
    config = calibrated_config(device_capacity_bytes=1 << 20)
    template = index_simulation_template(program, (), config)
    simulated = simulate_template(template, overlap_schedule())
    own = simulated.interval_arrays
    assert isinstance(own, IntervalArrays)

    # The way a store hands a result back: through its JSON, without arrays.
    read_back = _simulation_result_from_value(
        json.loads(json.dumps(asdict(simulated))), "stored"
    )
    assert read_back == simulated
    assert read_back.interval_arrays is None

    projected = interval_arrays_from_result(template, read_back)
    assert projected.task_interval_count == own.task_interval_count
    assert projected.transfer_interval_count == own.transfer_interval_count
    assert _rows(projected) == _rows(own)
