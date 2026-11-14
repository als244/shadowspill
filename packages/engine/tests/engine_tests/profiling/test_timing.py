from __future__ import annotations

from dataclasses import replace

import pytest

from shadowspill.profiling.timing import collect_timing_samples, condition_task
from shadowspill.task.profiling import ProfilingOptions, TimingWindow


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
    assert observed == TimingWindow(30, 300, 3, False, "wall")


def test_minimum_sample_count_survives_wall_cap() -> None:
    clock = Clock(device_ns=10, wall_ns=100)
    options = ProfilingOptions(
        minimum_samples=4,
        measurement_seconds=1,
        measurement_wall_seconds=50e-9,
    )
    measured = collect_timing_samples(clock.sample, options=options, clock=clock)
    assert measured.window == TimingWindow(40, 400, 4, False, "wall")
    assert measured.unstable(options)


def test_default_sampling_bounds_long_tasks_and_still_sustains_short_tasks() -> None:
    options = ProfilingOptions()
    long = Clock(device_ns=4_000_000_000, wall_ns=4_000_000_000)
    measured = collect_timing_samples(long.sample, options=options, clock=long)
    assert measured.window == TimingWindow(12_000_000_000, 12_000_000_000, 3, True)
    assert not measured.unstable(options)

    short = Clock(device_ns=1_000_000, wall_ns=1_000_000)
    measured = collect_timing_samples(short.sample, options=options, clock=short)
    assert measured.window.iterations == options.maximum_samples
    assert measured.window.limit == "samples"
    assert not measured.window.target_met
    assert not measured.unstable(options)


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
        "conditioning_max_samples",
        "minimum_samples",
        "maximum_samples",
    ],
)
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_iteration_count(name, value) -> None:
    with pytest.raises(ValueError, match=name):
        replace(ProfilingOptions(), **{name: value})


def test_count_limits_bound_tiny_tasks_without_claiming_duration_coverage():
    clock = Clock(device_ns=1_000, wall_ns=10_000)
    options = ProfilingOptions(
        conditioning_seconds=1,
        conditioning_max_samples=4,
        measurement_seconds=1,
        minimum_samples=3,
        maximum_samples=7,
    )
    warm = condition_task(clock.sample, options=options, clock=clock)
    assert warm == TimingWindow(4_000, 40_000, 4, False, "samples")
    measured = collect_timing_samples(clock.sample, options=options, clock=clock)
    assert measured.window == TimingWindow(7_000, 70_000, 7, False, "samples")
    assert not measured.unstable(options)
    assert clock.calls == 11


@pytest.mark.parametrize("duration", [-1, True, 1.0])
def test_invalid_sample_is_rejected(duration):
    with pytest.raises(ValueError, match="non-negative integer"):
        collect_timing_samples(
            lambda: duration,
            options=ProfilingOptions(minimum_samples=1),
        )


def test_sample_cap_preserves_instability_and_is_coordinated():
    clock = Clock()
    values = iter((100, 200, 300, 400))
    votes = []

    def sample():
        clock.sample()
        return next(values)

    def decide(needed, allowed):
        votes.append((needed, allowed))
        return needed and allowed

    options = ProfilingOptions(maximum_samples=4, measurement_seconds=1)
    timing = collect_timing_samples(
        sample, options=options, clock=clock, continue_while=decide
    )
    assert timing.window.limit == "samples"
    assert timing.unstable(options)
    assert votes == [(True, True), (True, False)]


def test_policy_refuses_conflicting_count_bounds_and_round_trips():
    with pytest.raises(ValueError, match="maximum_samples"):
        ProfilingOptions(minimum_samples=4, maximum_samples=3)
    options = ProfilingOptions(conditioning_max_samples=7, maximum_samples=13)
    assert ProfilingOptions.from_dict(options.to_dict()) == options
    window = TimingWindow(1, 2, 3, False, "samples")
    assert TimingWindow.from_dict(window.to_dict()) == window


def test_public_options_and_duration_shortcuts_have_one_effective_policy():
    from shadowspill import ProfilingOptions as PublicOptions
    from shadowspill.task.profiling import resolve_options

    assert PublicOptions is ProfilingOptions
    options = ProfilingOptions(conditioning_seconds=0.01, maximum_samples=9)
    assert resolve_options(options, None, None) is options
    assert resolve_options(None, 0.01, 0.02) == ProfilingOptions(
        conditioning_seconds=0.01, measurement_seconds=0.02
    )
    with pytest.raises(ValueError, match="not both"):
        resolve_options(options, 0.01, None)
    with pytest.raises(TypeError, match="ProfilingOptions"):
        resolve_options({}, None, None)
