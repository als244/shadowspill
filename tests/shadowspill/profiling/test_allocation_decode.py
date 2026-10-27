"""Logical requests are stable despite native retirement polling."""

import pytest

from shadowspill.errors import CaptureError
from shadowspill.profiling.observation.decode import decode_allocations
from shadowspill.task.allocations.events import TaskAllocationOperation


def event(action, address=4096, size=1024, stream=7):
    return dict(
        action=action,
        address=address,
        requested_bytes=size,
        stream=stream,
        pool_first=0,
        pool_second=0,
    )


def test_retired_same_stream_extents_reuse_one_lease():
    events = [
        event(action)
        for _ in range(16)
        for action in ("allocate", "free_requested", "free_completed")
    ]
    observed = decode_allocations(events, {}, ())
    allocations = [
        item
        for item in observed.trace
        if item.operation is TaskAllocationOperation.ALLOCATE
    ]
    assert [item.reuses_ordinal for item in allocations] == [None, *range(15)]
    assert observed.workspace_bytes == 1024
    assert observed.persistent_bytes == ()
    assert all(
        item.reuses_ordinal is None
        for item in observed.trace
        if item.operation is TaskAllocationOperation.FREE
    )


def test_native_cache_addresses_do_not_constrain_fixed_placement():
    observed = decode_allocations(
        [
            event("allocate", address=4096),
            event("free_requested", address=4096),
            event("allocate", address=8192),
            event("free_requested", address=8192),
        ],
        {},
        (),
    )
    assert observed.trace[2].reuses_ordinal == 0


@pytest.mark.parametrize("size,stream", [(512, 7), (2048, 7), (1024, 8)])
def test_different_extent_or_stream_does_not_imply_reuse(size, stream):
    observed = decode_allocations(
        [
            event("allocate"),
            event("free_requested"),
            event("allocate", size=size, stream=stream),
        ],
        {4096: [(0, 0)]},
        (),
    )
    assert observed.trace[-1].reuses_ordinal is None
    assert observed.trace[-1].output_leaf_indices == (0,)


def test_address_cannot_reappear_before_its_free_request():
    with pytest.raises(CaptureError, match="before its free request"):
        decode_allocations([event("allocate"), event("allocate")], {}, ())


def test_only_last_generation_publishes_output():
    observed = decode_allocations(
        [
            event("allocate"),
            event("free_requested"),
            event("allocate"),
            event("free_requested"),
        ],
        {4096: [(0, 16)]},
        (),
    )
    assert observed.trace[0].output_leaf_indices == ()
    assert observed.trace[2].output_leaf_indices == (0,)
    assert observed.trace[2].reuses_ordinal == 0


def test_repeated_temporary_fits_one_fixed_slot():
    from shadowspill.ir import ShadowSpillProgram, TaskProfile, TaskSpec
    from shadowspill.planner import SearchOptions
    from shadowspill.planner.admission.layout import build_fixed_layout_admission
    from shadowspill.planning.admission.facts import build_admission_facts
    from shadowspill.simulator import SimulationConfig
    from tests.shadowspill.planner._examples import COMPUTE, DEVICE

    observed = decode_allocations(
        [event(action) for _ in range(16) for action in ("allocate", "free_requested")],
        {},
        (),
    )
    program = ShadowSpillProgram(
        devices=(DEVICE,),
        alias_groups=(),
        objects=(),
        profiles=(TaskProfile("repeat", 10, 1024, "repeat"),),
        tasks=(TaskSpec("repeat", COMPUTE, "repeat"),),
    )
    facts = build_admission_facts(
        program,
        execution_pool_bytes=1024,
        object_capacity_bytes=1024,
        allocation_traces_by_compatibility={"repeat": observed.trace},
    )
    options = SearchOptions(workers=1)
    selected = options.resolved_algorithm(
        program,
        initial_residency=(),
        config=SimulationConfig.single_device(
            DEVICE.device_id,
            device_capacity_bytes=1024,
            spill_capacity_bytes=1024,
            fetch_solo_bandwidth_bytes_per_second=1,
            fetch_concurrent_bandwidth_bytes_per_second=1,
            evict_solo_bandwidth_bytes_per_second=1,
            evict_concurrent_bandwidth_bytes_per_second=1,
        ),
        generic=options.generic,
        workers=1,
        placement=facts,
    )
    certified = build_fixed_layout_admission(selected, facts)
    assert certified.layout.required_bytes == 1024


def test_retirement_notifications_do_not_change_contract_or_report_retention():
    from shadowspill.task.allocations.contract import TaskAllocationContract

    requests = [
        event("allocate", address=4096),
        event("free_requested", address=4096),
        event("allocate", address=8192),
        event("free_requested", address=8192),
    ]
    observations = [
        decode_allocations(events, {}, ())
        for events in (
            requests,
            [*requests[:2], event("free_completed"), *requests[2:]],
            [*requests, event("free_completed", address=8192), event("free_completed")],
        )
    ]
    assert (
        len(
            {
                TaskAllocationContract.capture(o.trace).compatibility_digest
                for o in observations
            }
        )
        == 1
    )
    assert all(o.trace == observations[0].trace for o in observations)
    assert all(o.persistent_bytes == () for o in observations)
    assert observations[0].trace[2].reuses_ordinal == 0
