import pytest

from shadowspill.runtime.failures import generic_runtime_error, read_allocator_failure
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
def test_failure_retains_its_destination_pool(pool_id):
    failure = read_allocator_failure(_EvictionFailure(pool_id), "task publication")
    assert failure is not None
    expected_pool = None if pool_id == (1 << 32) - 1 else pool_id
    assert failure.pool_id == expected_pool
    assert failure.as_dict()["pool_id"] == expected_pool
    error = generic_runtime_error(failure)
    assert error.diagnostics is failure
    assert "requested: 32\n" in str(error)
    assert "free: 16\n" in str(error)
    if expected_pool is None:
        assert "\npool:" not in str(error)
    else:
        assert f"\npool: {expected_pool}\n" in str(error)
