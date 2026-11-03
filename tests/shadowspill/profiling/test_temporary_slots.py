"""Different-sized temporaries reuse fixed addresses without sharing live storage."""

import pytest

from shadowspill.profiling.observation.decode import decode_allocations
from shadowspill.task.allocations.contract import TaskAllocationContract
from shadowspill.task.allocations.events import TaskAllocationOperation
from tests.shadowspill.profiling.test_allocation_decode import event


@pytest.mark.parametrize("sizes", [(256, 1024, 512), (1024, 256, 512)])
def test_sequential_sizes_share_one_slot_without_changing_request_contract(sizes):
    events = [
        event(action, address=4096 + index * 4096, size=size)
        for index, size in enumerate(sizes)
        for action in ("allocate", "free_requested")
    ]
    observed = decode_allocations(events, {}, ())
    allocated = [
        e for e in observed.trace if e.operation is TaskAllocationOperation.ALLOCATE
    ]
    assert [e.reuses_ordinal for e in allocated] == [None, 0, 1]
    assert [e.charged_bytes for e in allocated] == list(sizes)
    assert observed.workspace_bytes == max(sizes)
    contract = TaskAllocationContract.capture(observed.trace)
    assert [e.requested_bytes for e in contract.steps[::2]] == list(sizes)


def test_live_temporaries_and_distinct_streams_keep_distinct_slots():
    observed = decode_allocations(
        [
            event("allocate", address=4096, size=1024),
            event("allocate", address=8192, size=512),
            event("free_requested", address=8192),
            event("free_requested", address=4096),
            event("allocate", address=12288, size=256, stream=8),
            event("free_requested", address=12288),
        ],
        {},
        (),
    )
    assert all(e.reuses_ordinal is None for e in observed.trace)
    assert observed.workspace_bytes == 1536


def test_returned_view_does_not_inherit_a_larger_temporary_slot():
    observed = decode_allocations(
        [
            event("allocate", address=4096, size=1024),
            event("free_requested", address=4096),
            event("allocate", address=8192, size=256),
        ],
        {8192: [(0, 128)]},
        (),
    )
    assert observed.trace[-1].reuses_ordinal is None
    assert observed.trace[-1].output_view_offsets == (128,)


def test_interval_assignment_can_fill_gaps_between_large_requests():
    observed = decode_allocations(
        [
            event(action, address=4096 + index * 4096, size=size)
            for index, size in enumerate((1024, 256, 1024, 512))
            for action in ("allocate", "free_requested")
        ],
        {},
        (),
    )
    allocated = [
        e for e in observed.trace if e.operation is TaskAllocationOperation.ALLOCATE
    ]
    assert [e.reuses_ordinal for e in allocated] == [None, 0, 1, 2]
