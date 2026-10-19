"""Task resource timing must retain queue order and isolated whole-task time."""

from dataclasses import replace

import pytest

from shadowspill.errors import ProfilingError
from shadowspill.ir import TaskResourcePhase
from shadowspill.pytorch.profiling.profiler.collectives import resource_phases
from shadowspill.task.profiles import ProfileEnvironment, ProfileKey


def sample(start=2.0, end=5.0, duration=10.0, group="weights"):
    return {
        "duration_ms": duration,
        "errors": 0,
        "records": [
            {
                "start_ms": start,
                "end_ms": end,
                "operation": "_allgather_base_",
                "group": group,
            }
        ],
    }


def test_only_collective_interval_reserves_shared_resource():
    result = resource_phases(
        [sample(), sample(2.2, 5.2), sample(1.8, 4.8)],
        resources={"weights": "sharded-collectives"},
        runtime_ns=9_000_000,
    )
    assert result == (
        TaskResourcePhase(1_800_000),
        TaskResourcePhase(2_700_000, "sharded-collectives"),
        TaskResourcePhase(4_500_000),
    )


def test_normalization_and_rounding_preserve_exact_task_duration():
    result = resource_phases(
        [sample()], resources={"weights": "queue"}, runtime_ns=10_000_001
    )
    assert sum(phase.duration_ns for phase in result) == 10_000_001


def test_no_collectives_leaves_ordinary_compute_profile():
    assert not resource_phases(
        [{"duration_ms": 2, "records": []}], resources={}, runtime_ns=1000
    )


@pytest.mark.parametrize("error", [1, 4])
def test_missing_or_dropped_events_are_not_accepted(error):
    with pytest.raises(ProfilingError, match="incomplete"):
        resource_phases(
            [{**sample(), "errors": error}],
            resources={"weights": "queue"},
            runtime_ns=10_000_000,
        )


def test_unregistered_groups_are_not_merged_into_background_queue():
    with pytest.raises(ProfilingError, match="unregistered"):
        resource_phases([sample()], resources={}, runtime_ns=10_000_000)


def test_changed_collective_sequence_is_rejected():
    with pytest.raises(ProfilingError, match="sequence changed"):
        resource_phases(
            [sample(), sample(group="experts")],
            resources={"weights": "queue-a", "experts": "queue-b"},
            runtime_ns=10_000_000,
        )


def test_overlapping_communicators_are_not_flattened():
    value = sample()
    value["records"] += sample(4.0, 6.0, group="experts")["records"]
    with pytest.raises(ProfilingError, match="overlapping"):
        resource_phases(
            [value],
            resources={"weights": "queue-a", "experts": "queue-b"},
            runtime_ns=10_000_000,
        )


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan"), True])
def test_bad_event_times_are_rejected(value):
    with pytest.raises(ProfilingError, match="finite nonnegative"):
        resource_phases(
            [sample(start=value)],
            resources={"weights": "queue"},
            runtime_ns=10_000_000,
        )


def test_instrumentation_overhead_cannot_hide_unstable_task_timing():
    with pytest.raises(ProfilingError, match="unstable profile"):
        resource_phases(
            [sample()], resources={"weights": "queue"}, runtime_ns=1_000_000
        )


def test_new_measurement_policy_cannot_reuse_old_profile_cache():
    environment = ProfileEnvironment("torch", None, "gpu", (9, 0), "compiler", "pool")
    assert ProfileKey("a" * 64, environment).digest != ProfileKey(
        "a" * 64,
        replace(
            environment,
            timing_policy="batched-events+task-collective-phases+paired-progress/v2",
        ),
    ).digest
