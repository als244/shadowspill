"""Shared startup costs cannot mix component maxima under a smaller total."""

from types import SimpleNamespace

import pytest

from shadowspill.communication import CommunicationInfo
from shadowspill.planning.worker_memory import (
    communicator_reservation,
    external_baseline,
    shared_fixed_overhead,
)
from shadowspill.resources import Resource


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
        pool_memory=None,
        profiled_program=SimpleNamespace(
            fixed_device_bytes=external + headroom + retained,
            metadata=dict(
                external_device_bytes=external,
                external_headroom_bytes=headroom,
                stream_workspace_bytes=retained,
                resource_streams=[{"lane": 0}],
            )
            if metadata
            else {},
        ),
        program=SimpleNamespace(
            parallelism=SimpleNamespace(world_size=2) if distributed else None,
            resources={},
            bindings={},
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
    assert shared_fixed_overhead([legacy, legacy]) == 80
    assert shared_fixed_overhead([legacy, plan(8, 2, 10)]) == 60


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


def resource_plan(name, size):
    value = plan(8, 2, 10 + size)
    value.profiled_program.metadata.update(
        stream_workspace_bytes=10, provider_resource_bytes=size
    )
    value.program.resources = {name: Resource("not_imported.provider:factory")}
    value.program.bindings = {"resource_sources": {name: "a" * 64}}
    return value


def test_shared_worker_reserves_distinct_provider_inventories_and_deduplicates_exact():
    first, second = resource_plan("first", 100), resource_plan("second", 200)
    # Context8 + margin2 + shared streams10 + both simultaneously live resources.
    assert shared_fixed_overhead([first, second, first]) == 320
    assert shared_fixed_overhead([first, first]) == 120


def test_conflicting_provider_descriptions_fail_during_cpu_admission():
    first, second = resource_plan("shared", 100), resource_plan("shared", 100)
    second.program.resources["shared"] = Resource(
        "not_imported.provider:factory", {"capacity": 2}
    )
    with pytest.raises(ValueError, match="conflicting descriptions"):
        shared_fixed_overhead([first, second])


def test_prepared_bindings_stream_sets_and_private_retained_state_are_additive():
    first, second = plan(8, 2, 40), plan(8, 2, 60)
    for value, lane, size, source in ((first, 0, 20, "a"), (second, 1, 30, "b")):
        value.profiled_program.metadata.update(
            stream_workspace_bytes=10,
            resource_streams=[{"lane": lane}],
            prepared_operator_retained_bytes=size,
        )
        value.program.bindings["operator_preparation"] = {"source": source}
    # Two stream inventories, two binary bindings, 10+20 private retained bytes.
    assert shared_fixed_overhead([first, second]) == 110
    # Each path restores functions independently; private storage is not deduped.
    assert shared_fixed_overhead([first, first]) == 60


def test_retained_components_require_inventory_and_fit_the_observed_total():
    value = plan(retained=10)
    value.profiled_program.metadata["provider_resource_bytes"] = 1
    with pytest.raises(ValueError, match="lacks its retained inventory"):
        shared_fixed_overhead([value])
    value = resource_plan("provider", 100)
    value.profiled_program.metadata["provider_resource_bytes"] = 101
    with pytest.raises(ValueError, match="components exceed"):
        shared_fixed_overhead([value])
