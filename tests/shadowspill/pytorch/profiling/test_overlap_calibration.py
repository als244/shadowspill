"""Interference evidence is independent of a full-step fit or copy tail stalls."""

from dataclasses import replace

import pytest
import torch

from shadowspill.pytorch.distributed._backing_storage import collective_timing_class
from shadowspill.pytorch.profiling.profiler.overlap import (
    PairObservation,
    _probe_work_counts,
    paired_progress,
)


def trial():
    task = PairObservation(0, 100, ())
    copies = PairObservation(0, 0, ((0, 20), (20, 40), (40, 60)))
    pair = PairObservation(0, 140, ((20, 100), (100, 140), (140, 210)))
    return task, copies, pair


def test_only_actual_overlap_and_interior_copies_determine_progress():
    task, copies, pair = trial()
    rates = paired_progress([task] * 5, [copies] * 5, [pair] * 5)
    assert rates == pytest.approx((2 / 3, 0.5))


def test_arbitrarily_long_post_task_stall_does_not_change_progress():
    task, copies, pair = trial()
    delayed = replace(pair, transfers=(*pair.transfers[:-1], (140, 1000000)))
    assert paired_progress([task] * 5, [copies] * 5, [delayed] * 5) == pytest.approx(
        (2 / 3, 0.5)
    )


def test_no_interior_copy_is_unidentifiable_not_a_fabricated_rate():
    task, copies, pair = trial()
    partial = replace(pair, transfers=((20, 100), (100, 210), (210, 300)))
    assert paired_progress([task] * 5, [copies] * 5, [partial] * 5) is None


def test_nonphysical_trial_is_discarded_without_rate_floor():
    task, copies, pair = trial()
    invalid = replace(pair, task_end_ns=1000)
    assert paired_progress(
        [task] * 5, [copies] * 5, [invalid, pair, pair, pair, pair]
    ) == pytest.approx((2 / 3, 0.5))
    assert paired_progress([task] * 5, [copies] * 5, [invalid] * 5) is None


def test_at_least_three_trials_are_needed():
    task, copies, pair = trial()
    assert paired_progress([task] * 2, [copies] * 2, [pair] * 2) is None


@pytest.mark.parametrize(
    "task_ns,copy_ns", [(5_500_000, 2_100_000), (15_000_000, 4_100_000)]
)
def test_large_copy_probes_cover_multiple_collectives(task_ns, copy_ns):
    tasks, copies = _probe_work_counts(task_ns, copy_ns)
    assert tasks > 1
    assert tasks * task_ns >= 4 * copy_ns
    assert copies * copy_ns >= 2 * tasks * task_ns


def test_probe_work_is_bounded_at_extreme_size_ratios():
    assert _probe_work_counts(1, 1_000_000) == (8, 3)
    assert _probe_work_counts(1_000_000, 1) == (1, 128)


def test_repetition_can_identify_an_otherwise_unobservable_copy():
    # One 5-unit task ends before a full later copy. A four-call observation
    # includes completed copies without borrowing their post-compute tails.
    task = PairObservation(0, 20, ())
    copies = PairObservation(0, 0, tuple((i * 4, (i + 1) * 4) for i in range(10)))
    pair = PairObservation(2, 30, tuple((i * 8, (i + 1) * 8) for i in range(10)))
    assert paired_progress([task] * 5, [copies] * 5, [pair] * 5) == pytest.approx(
        (5 / 7, 1 / 2)
    )


def test_timing_classes_distinguish_operation_dtype_and_group_geometry():
    classes = {
        collective_timing_class("gather", 4, torch.uint8),
        collective_timing_class("gather", 8, torch.uint8),
        collective_timing_class("reduce", 8, torch.float32),
        collective_timing_class("reduce", 8, torch.bfloat16),
        collective_timing_class("reduce", 4, torch.float32, 2),
    }
    assert len(classes) == 5
