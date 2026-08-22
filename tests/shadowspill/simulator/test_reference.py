from __future__ import annotations

import json
from pathlib import Path

from reference.python.simulator import simulate_python
from shadowspill.simulator import TransferDirection, simulate
from tests.shadowspill.ir._examples import (
    SAVE_SELECTION,
    representative_program,
    representative_schedule,
)

from ._examples import (
    calibrated_config,
    concurrent_lane_program,
    initial_only_schedule,
    ordered_action_program,
    ordered_action_schedule,
    overlap_program,
    overlap_schedule,
)


def test_quick_turnaround_waits_for_evict_then_fetch() -> None:
    result = simulate_python(
        representative_program(),
        representative_schedule(),
        selections=SAVE_SELECTION,
        config=calibrated_config(device_capacity_bytes=600),
        record_timeline=True,
    )

    assert result.makespan_ns == 278
    assert tuple(
        (item.direction, item.start_ns, item.end_ns)
        for item in result.transfer_intervals
    ) == (
        (TransferDirection.EVICT, 10, 138),
        (TransferDirection.FETCH, 138, 266),
    )
    consume = result.task_intervals[-1]
    assert consume.task_id == "consume"
    assert consume.ready_ns == 10
    assert consume.start_ns == 266
    assert consume.stall_ns == 256
    assert consume.stall_reasons == ("input-residency",)
    peak = result.device_peak("cuda_0")
    assert peak.object_bytes == 512
    assert peak.workspace_bytes == 16
    assert peak.total_bytes == 528
    assert result.spill_peak_bytes == 384
    assert result.memory_timeline
    assert result.transfer_intervals[1].stall_reasons == ("source-readiness",)
    assert result.transfer_intervals[1].stall_ns == 128


def test_prefetch_overlaps_unrelated_compute() -> None:
    result = simulate(
        overlap_program(),
        overlap_schedule(),
        config=calibrated_config(device_capacity_bytes=512),
    )

    assert result.makespan_ns == 900
    evict, fetch = result.transfer_intervals
    assert (evict.start_ns, evict.end_ns) == (100, 228)
    assert (fetch.start_ns, fetch.end_ns) == (500, 628)
    spacer = result.task_intervals[2]
    assert (spacer.start_ns, spacer.end_ns) == (500, 800)
    assert fetch.start_ns == spacer.start_ns
    assert fetch.end_ns < spacer.end_ns
    consume = result.task_intervals[3]
    assert consume.start_ns == 800
    assert consume.stall_ns == 0


def test_overlap_case_matches_frozen_external_oracle_artifact() -> None:
    result = simulate(
        overlap_program(),
        overlap_schedule(),
        config=calibrated_config(device_capacity_bytes=512),
    )
    root = Path(__file__).resolve().parents[3]
    artifact = json.loads(
        (root / "tests/fixtures/simulator/reference_v1.json").read_text()
    )

    assert artifact["makespan_ns"] == result.makespan_ns
    assert artifact["device_peak_bytes"] == result.device_peak("cuda_0").total_bytes
    assert artifact["spill_peak_bytes"] == result.spill_peak_bytes


def test_distinct_resource_lanes_execute_concurrently() -> None:
    result = simulate(
        concurrent_lane_program(),
        initial_only_schedule(),
        config=calibrated_config(),
    )

    assert result.makespan_ns == 200
    assert tuple(
        (item.task_id, item.start_ns, item.end_ns) for item in result.task_intervals
    ) == (
        ("compute", 0, 100),
        ("communication", 0, 200),
    )
    peak = result.device_peak("cuda_0")
    assert peak.object_bytes == 64
    assert peak.workspace_bytes == 12
    assert peak.total_bytes == 76


def test_result_is_deterministic_across_replays() -> None:
    arguments = (
        overlap_program(),
        overlap_schedule(),
    )
    config = calibrated_config(device_capacity_bytes=512)

    first = simulate(*arguments, config=config)
    second = simulate(*arguments, config=config)

    assert second == first


def test_actions_preserve_plan_order_across_concurrent_task_completion() -> None:
    result = simulate(
        ordered_action_program(),
        ordered_action_schedule(),
        config=calibrated_config(),
    )

    assert tuple(item.trigger_task_id for item in result.transfer_intervals) == (
        "long_task",
        "short_task",
    )
    assert tuple(
        (item.start_ns, item.end_ns) for item in result.transfer_intervals
    ) == ((200, 264), (264, 296))
