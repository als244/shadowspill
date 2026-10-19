from dataclasses import replace

import pytest

from shadowspill.runtime.failures import allocator_oom_error, read_allocator_failure
from shadowspill.status import Status


class _EvictionFailure:
    def __init__(self, pool_id):
        self.pool_id = pool_id

    def shadowspill_pytorch_allocator_failure(self, pointer):
        failure = pointer._obj
        failure.device_ordinal = 0
        failure.runtime.pool_id = self.pool_id
        failure.runtime.requested_bytes = 32
        failure.runtime.free_bytes = 16
        failure.runtime.largest_free_range_bytes = 16
        return Status.NO_PROGRESS


@pytest.mark.parametrize("pool_id", [0, 1, (1 << 32) - 1])
def test_failure_retains_destination_pool_and_refused_eviction(pool_id):
    failure = read_allocator_failure(_EvictionFailure(pool_id), "task publication")
    assert failure is not None
    expected_pool = None if pool_id == (1 << 32) - 1 else pool_id
    assert failure.pool_id == expected_pool
    assert failure.as_dict()["pool_id"] == expected_pool
    failure = replace(
        failure,
        refused_action="EVICT activation (trigger task_28)",
        object_state="execution_ready, spill_current=0",
    )
    error = allocator_oom_error(failure)
    assert error.diagnostics is failure
    assert failure.as_dict()["refused_action"] == failure.refused_action
    assert failure.as_dict()["object_state"] == failure.object_state
    assert "refused_action: EVICT activation (trigger task_28)" in str(error)
    assert "object_state: execution_ready, spill_current=0" in str(error)
    assert "requested: 32 B" in str(error)
    assert "pool free: 16 B" in str(error)
    if expected_pool is None:
        assert "\npool:" not in str(error)
    else:
        assert f"\npool: {expected_pool}\n" in str(error)
