"""Measured interference must not incorporate rank-arrival or post-task stalls."""

from dataclasses import replace

import pytest

from shadowspill.profiling.overlap_statistics import (
    PairObservation,
    paired_progress,
    probe_work_counts,
)


def trial():
    return (
        PairObservation(0, 100, ()),
        PairObservation(0, 0, ((0, 20), (20, 40), (40, 60))),
        PairObservation(0, 140, ((20, 100), (100, 140), (140, 210))),
    )


def test_measured_rates_use_actual_overlap_and_interior_transfers():
    task, copies, pair = trial()
    assert paired_progress([task] * 5, [copies] * 5, [pair] * 5) == pytest.approx(
        (2 / 3, 0.5)
    )
    delayed = replace(pair, transfers=(*pair.transfers[:-1], (140, 1e9)))
    assert paired_progress([task] * 5, [copies] * 5, [delayed] * 5) == pytest.approx(
        (2 / 3, 0.5)
    )


def test_missing_interior_and_nonphysical_observations_are_not_guessed():
    task, copies, pair = trial()
    partial = replace(pair, transfers=((20, 100), (100, 210), (210, 300)))
    invalid = replace(pair, task_end_ns=1000)
    for value in (partial, invalid):
        assert paired_progress([task] * 5, [copies] * 5, [value] * 5) is None
    assert paired_progress(
        [task] * 5, [copies] * 5, [invalid, pair, pair, pair, pair]
    ) == pytest.approx((2 / 3, 0.5))
    assert paired_progress([task] * 2, [copies] * 2, [pair] * 2) is None


def test_work_counts_bound_tiny_tasks_and_very_large_messages():
    for runtime, transfer in ((1, 1), (10**9, 1), (1, 10**9), (2_000_000, 1_000_000)):
        tasks, copies = probe_work_counts(runtime, transfer)
        assert 1 <= tasks <= 8 and 3 <= copies <= 128
