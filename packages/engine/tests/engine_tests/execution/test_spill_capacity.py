"""Execution reserves the selected worker requirement, not its search ceiling."""

import pytest

from shadowspill.memory import PinnedHostPool, SpillPool
from shadowspill.planning.spill import spill_configuration


@pytest.mark.parametrize(
    "required,budget,capacity",
    [
        (100, 1000, 105),
        (101, 1000, 107),
        (950, 1000, 998),
        (975, 1000, 1000),
        (1000, 1000, 1000),
        (0, 1000, 1),
    ],
)
def test_default_uses_five_percent_capped_by_budget(required, budget, capacity):
    pool = spill_configuration(required, budget, None)
    assert type(pool) is PinnedHostPool
    assert pool.capacity == capacity


def test_explicit_pool_is_preserved_and_must_cover_the_whole_worker():
    pool = SpillPool(150)
    assert spill_configuration(140, 1000, pool) is pool
    with pytest.raises(ValueError, match="cover the worker"):
        spill_configuration(160, 1000, pool)
    with pytest.raises(ValueError, match="exceeding the Plan"):
        spill_configuration(100, 140, pool)


def test_only_declared_backends_reuse_larger_capacity_memory_evidence():
    pinned, extension = PinnedHostPool(100), SpillPool(100)
    assert pinned.accepts_memory_profile_capacity(1000)
    assert not extension.accepts_memory_profile_capacity(1000)
    assert extension.accepts_memory_profile_capacity(100)
    assert not pinned.accepts_memory_profile_capacity(99)
