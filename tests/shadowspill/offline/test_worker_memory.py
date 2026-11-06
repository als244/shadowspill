"""Shared startup costs cannot mix component maxima under a smaller total."""

from types import SimpleNamespace

import pytest

from shadowspill.communication import CommunicationInfo
from shadowspill.planning.worker_memory import (
    communicator_reservation,
    external_baseline,
    shared_fixed_overhead,
)


def plan(
    external=0,
    headroom=0,
    retained=0,
    *,
    communication=None,
    scratch=0,
    distributed=False,
    metadata=True,
    active_collectives=True,
):
    info = communication or CommunicationInfo(1e10, 1e10, fixed_device_bytes=0)
    return SimpleNamespace(
        profiled_program=SimpleNamespace(
            fixed_device_bytes=external + headroom + retained,
            metadata=dict(
                external_device_bytes=external, external_headroom_bytes=headroom
            )
            if metadata
            else {},
        ),
        program=SimpleNamespace(
            parallelism=SimpleNamespace(world_size=2) if distributed else None
        ),
        communication_info=info,
        _communication_memory=(
            (info.fixed_device_bytes or 0) + scratch if active_collectives else 0,
            (),
        ),
    )


def test_crossed_external_and_headroom_maxima_are_both_reserved():
    plans = [plan(128, 768, 64), plan(768, 128, 64)]
    assert external_baseline(plans) == 768
    assert shared_fixed_overhead(plans) == 1600
    assert shared_fixed_overhead(plans * 2) == 1600


def test_retained_component_is_separate_from_the_external_maximum():
    plans = [plan(500, 100, 20), plan(100, 100, 500)]
    assert shared_fixed_overhead(plans) == 1100
    assert external_baseline(plans) == 500


def test_one_dominating_plan_keeps_its_existing_total():
    assert shared_fixed_overhead([plan(10, 20, 30), plan(8, 19, 25)]) == 60


def test_legacy_undecomposed_cost_is_not_discarded_or_guessed():
    legacy = plan(retained=40, metadata=False)
    assert shared_fixed_overhead([legacy, legacy]) == 40
    assert shared_fixed_overhead([legacy, plan(8, 2, 10)]) == 50


def test_distinct_communicator_startup_is_summed_but_not_charged_twice():
    first = CommunicationInfo(1e10, 1e10, fixed_device_bytes=4)
    second = CommunicationInfo(1e10, 1e10, fixed_device_bytes=6)
    plans = [
        plan(8, 2, 10, communication=first, scratch=2, distributed=True),
        plan(2, 8, 10, communication=second, scratch=3, distributed=True),
    ]
    assert communicator_reservation(plans) == 10
    # external8 + headroom8 + retained/scratch13 + communicators10.
    assert shared_fixed_overhead(plans) == 39
    assert shared_fixed_overhead(plans * 2) == 39


def test_equivalent_communication_records_share_one_startup_reservation():
    first = CommunicationInfo(1e10, 1e10, fixed_device_bytes=4)
    second = CommunicationInfo.from_dict(first.to_dict())
    plans = [
        plan(8, 2, 10, communication=first, scratch=2, distributed=True),
        plan(2, 8, 10, communication=second, scratch=3, distributed=True),
    ]
    assert communicator_reservation(plans) == 4
    assert shared_fixed_overhead(plans) == 33


def test_independent_path_keeps_the_distributed_worker_reservation():
    first = CommunicationInfo(1e10, 1e10, fixed_device_bytes=4)
    second = CommunicationInfo(1e10, 1e10, fixed_device_bytes=6)
    plans = [
        plan(8, 2, 10, communication=first, scratch=2, distributed=True),
        plan(2, 8, 10, communication=second, active_collectives=False),
    ]
    assert communicator_reservation(plans) == 10
    assert shared_fixed_overhead(plans) == 38


def test_non_dp_communicator_memory_is_preserved_without_a_new_runtime_rule():
    info = CommunicationInfo(1e10, 1e10, fixed_device_bytes=4)
    plans = [plan(8, 2, 10, communication=info, scratch=3)]
    assert communicator_reservation(plans) == 0
    assert shared_fixed_overhead(plans) == 27


def test_invalid_split_is_not_hidden_by_a_nonnegative_clamp():
    invalid = plan(8, 2, 10)
    invalid.profiled_program.fixed_device_bytes = 9
    with pytest.raises(ValueError, match="baseline and headroom"):
        shared_fixed_overhead([invalid])
