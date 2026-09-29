from __future__ import annotations

from dataclasses import replace

import pytest

from shadowspill.profiling.timing import collect_timing_samples, condition_task
from shadowspill.pytorch import ProfilingOptions
from shadowspill.task.profiles import TaskMeasurement
from shadowspill.task.profiling import TimingWindow


class Clock:
    def __init__(self, device_ns: int = 100, wall_ns: int = 100) -> None:
        self.device_ns = device_ns
        self.wall_ns = wall_ns
        self.now = 0
        self.calls = 0

    def sample(self) -> int:
        self.now += self.wall_ns
        self.calls += 1
        return self.device_ns

    def __call__(self) -> int:
        return self.now


def test_conditioning_uses_device_time_and_excludes_measurement_samples() -> None:
    clock = Clock(device_ns=100, wall_ns=200)
    options = ProfilingOptions(
        conditioning_seconds=500e-9,
        conditioning_wall_seconds=2_000e-9,
        minimum_samples=3,
        measurement_seconds=400e-9,
        measurement_wall_seconds=2_000e-9,
    )
    warm = condition_task(clock.sample, options=options, clock=clock)
    measured = collect_timing_samples(clock.sample, options=options, clock=clock)
    assert warm == TimingWindow(500, 1_000, 5, True)
    assert measured.window == TimingWindow(400, 800, 4, True)
    assert measured.samples == (100,) * 4
    assert not measured.unstable(options)
    assert clock.calls == 9


def test_conditioning_wall_cap_reports_unreached_target() -> None:
    clock = Clock(device_ns=10, wall_ns=100)
    options = ProfilingOptions(
        conditioning_seconds=1,
        conditioning_wall_seconds=250e-9,
    )
    observed = condition_task(clock.sample, options=options, clock=clock)
    # Checks occur between invocations; the final invocation can cross the cap.
    assert observed == TimingWindow(30, 300, 3, False)


def test_minimum_sample_count_survives_wall_cap() -> None:
    clock = Clock(device_ns=10, wall_ns=100)
    options = ProfilingOptions(
        minimum_samples=4,
        measurement_seconds=1,
        measurement_wall_seconds=50e-9,
    )
    measured = collect_timing_samples(clock.sample, options=options, clock=clock)
    assert measured.window == TimingWindow(40, 400, 4, False)
    assert measured.unstable(options)


def test_zero_device_times_are_bounded_by_wall_time() -> None:
    clock = Clock(device_ns=0, wall_ns=100)
    options = ProfilingOptions(
        conditioning_seconds=1,
        conditioning_wall_seconds=200e-9,
        minimum_samples=3,
        measurement_seconds=1,
        measurement_wall_seconds=200e-9,
    )
    assert not condition_task(clock.sample, options=options, clock=clock).target_met
    measured = collect_timing_samples(clock.sample, options=options, clock=clock)
    assert measured.samples == (0, 0, 0)
    assert measured.unstable(options)


def test_duration_floors_can_be_disabled() -> None:
    clock = Clock()
    options = ProfilingOptions(
        conditioning_seconds=0,
        minimum_samples=2,
        measurement_seconds=0,
    )
    assert condition_task(clock.sample, options=options, clock=clock) == TimingWindow()
    measured = collect_timing_samples(clock.sample, options=options, clock=clock)
    assert measured.samples == (100, 100)
    assert clock.calls == 2


def test_stability_thresholds_are_configurable() -> None:
    def measure(threshold: float):
        clock = Clock()
        values = iter((100, 100, 200, 200))

        def sample():
            clock.sample()
            return next(values)

        options = ProfilingOptions(
            minimum_samples=4,
            measurement_seconds=0,
            measurement_wall_seconds=400e-9,
            relative_mad_threshold=threshold,
            half_drift_threshold=threshold,
        )
        return collect_timing_samples(sample, options=options, clock=clock), options

    strict, policy = measure(0.03)
    assert strict.unstable(policy)
    loose, policy = measure(1.0)
    assert not loose.unstable(policy)


@pytest.mark.parametrize(
    "name",
    [
        "conditioning_seconds",
        "conditioning_wall_seconds",
        "measurement_seconds",
        "measurement_wall_seconds",
        "relative_mad_threshold",
        "half_drift_threshold",
    ],
)
@pytest.mark.parametrize("value", [-1, float("inf"), float("nan"), True])
def test_invalid_duration_or_threshold(name, value) -> None:
    with pytest.raises(ValueError, match=name):
        replace(ProfilingOptions(), **{name: value})


@pytest.mark.parametrize(
    "name",
    [
        "warmup_iterations",
        "minimum_samples",
        "stabilization_iterations",
    ],
)
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_iteration_count(name, value) -> None:
    with pytest.raises(ValueError, match=name):
        replace(ProfilingOptions(), **{name: value})


def test_measurement_persists_policy_and_timing_evidence() -> None:
    record = TaskMeasurement(
        100,
        0,
        0,
        (),
        (100,),
        "test",
        profiling_options=ProfilingOptions(conditioning_seconds=0.5),
        conditioning=TimingWindow(500_000_000, 600_000_000, 10, True),
        sampling=TimingWindow(100, 200, 1, False),
        timing_unstable=True,
    )
    assert TaskMeasurement.from_dict(record.to_dict()) == record
