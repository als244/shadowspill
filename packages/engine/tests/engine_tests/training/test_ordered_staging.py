"""Later uploads cannot starve the earlier call that blocks their acceptance."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from shadowspill.orchestration.launch.admission import NodeAdmission, NodeCapacity
from shadowspill.orchestration.launch.host_arena import HostArena
from shadowspill.training.session.staging import OrderedStaging
from shadowspill_client import Execution
from shadowspill_client.buffers import BufferBudget


def queue(*, size=10, count=2):
    return OrderedStaging(Execution(max_staging_bytes=size, max_pending_requests=count))


def test_input_owners_retain_node_arena_charge_through_queue_shutdown():
    node = NodeAdmission(NodeCapacity("controller", (), 100))
    arena = HostArena(node, 10, 2)
    inputs = OrderedStaging(
        Execution(max_staging_bytes=10, max_pending_requests=2), budget=arena.budget
    )
    first = inputs.register(1, 8)
    payload = first.acquire().reserve(8)
    assert arena.budget.used == inputs.used == (8, 1)
    inputs.close()
    with pytest.raises(RuntimeError, match="live buffer owners"):
        arena.close()
    assert node.used.host_bytes == 10
    with pytest.raises(RuntimeError, match="live input owners"):
        first.close()
    payload.close()
    first.close()
    arena.close()
    assert node.used.claims == 0


@pytest.mark.parametrize(
    "size,count,used", [(9, 2, False), (10, 1, False), (10, 2, True)]
)
def test_parent_budget_must_match_configured_input_capacity_and_have_no_owner(
    size, count, used
):
    budget = BufferBudget(size, count)
    owner = budget.reserve(0) if used else None
    try:
        with pytest.raises(ValueError, match="matching Execution"):
            OrderedStaging(
                Execution(max_staging_bytes=10, max_pending_requests=2), budget=budget
            )
    finally:
        if owner is not None:
            owner.close()


def test_out_of_order_metadata_reserves_earlier_pipeline_before_later_upload():
    inputs = queue()
    later = inputs.register(2, 8)
    assert inputs.used == (0, 0)
    with pytest.raises(TimeoutError):
        later.acquire(timeout=0)
    earlier = inputs.register(1, 10)
    owned = earlier.acquire(timeout=0)
    assert inputs.used == (10, 1)
    with ThreadPoolExecutor(max_workers=1) as threads:
        pending = threads.submit(later.acquire, timeout=2)
        inner = owned.reserve(10)
        with pytest.raises(RuntimeError, match="live input owners"):
            earlier.close()
        assert inputs.used == (10, 1) and not pending.done()
        inner.close()
        earlier.close()
        assert pending.result(timeout=2).max_bytes == 8
    assert inputs.used == (8, 1)
    later.close()
    assert inputs.used == (0, 0)


def test_full_window_backpressure_keeps_room_for_missing_earlier_metadata():
    inputs = queue()
    later = inputs.register(2, 4)
    with pytest.raises(TimeoutError, match="window is full"):
        inputs.register(3, 4)
    first = inputs.register(1, 4)
    assert inputs.used == (8, 2)
    first.close()
    third = inputs.register(3, 4)
    assert inputs.used == (8, 2)
    later.close()
    third.close()
    assert inputs.used == (0, 0)


def test_granted_uploads_can_finish_out_of_order_without_recycling_sequence_ids():
    inputs = queue()
    first, second = inputs.register(1, 4), inputs.register(2, 4)
    second.acquire(timeout=0)
    second.close()
    with pytest.raises(ValueError, match="already owns or consumed"):
        inputs.register(2, 4)
    first.acquire(timeout=0)
    first.close()
    third = inputs.register(3, 8)
    third.acquire(timeout=0)
    third.close()


def test_rejected_waiting_tickets_release_order_without_owning_input_bytes():
    inputs = queue(count=3)
    third = inputs.register(3, 8)
    second = inputs.register(2, 8)
    second.close()
    first = inputs.register(1, 10)
    first.close()
    assert third.acquire(timeout=0).max_bytes == 8
    third.close()
    assert inputs.used == (0, 0)


def test_queue_shutdown_wakes_waiters_and_preserves_claimed_storage():
    inputs = queue()
    first, second = inputs.register(1, 10), inputs.register(2, 8)
    owned = first.acquire(timeout=0)
    inner = owned.reserve(4)
    with ThreadPoolExecutor(max_workers=1) as threads:
        pending = threads.submit(second.acquire, timeout=2)
        inputs.close()
        with pytest.raises(RuntimeError, match="closed"):
            pending.result(timeout=2)
    assert inputs.used == (10, 1)
    inner.close()
    first.close()
    assert inputs.used == (0, 0)
    with pytest.raises(RuntimeError, match="closed"):
        inputs.register(3, 1)


def test_unclaimed_grants_are_released_on_shutdown_and_ownership_transfers_once():
    inputs = queue()
    ticket = inputs.register(1, 4)
    ticket.acquire(timeout=0)
    with pytest.raises(RuntimeError, match="already has an owner"):
        ticket.acquire(timeout=0)
    unused = inputs.register(2, 4)
    inputs.close()
    assert inputs.used == (4, 1)
    with pytest.raises(RuntimeError, match="closed"):
        unused.acquire(timeout=0)
    ticket.close()
    ticket.close()
    assert inputs.used == (0, 0)


def test_advertised_pipeline_bound_is_checked_before_queue_admission():
    inputs = queue()
    with pytest.raises(ValueError, match="exceeds"):
        inputs.register(1, 11)
    assert inputs.used == (0, 0)
    ticket = inputs.register(1, 4)
    arena = ticket.acquire(timeout=0)
    with pytest.raises(ValueError, match="limit is 4"):
        arena.reserve(5)
    ticket.close()
