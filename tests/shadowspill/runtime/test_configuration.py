"""What a caller asks of the runtime, checked: pools and routes, budgets, the device."""

from __future__ import annotations

import ctypes
from pathlib import Path

import pytest

from shadowspill.errors import AdmissionError
from shadowspill.memory import (
    PinnedHostPool,
    SpillPool,
    device,
    pinned_host,
    transfer_route,
)
from shadowspill.runtime import MemoryPool
from shadowspill.runtime.configuration import (
    RuntimeConfigurationError,
    Topology,
    resolve_dynamic_scratch_reserve,
    resolve_execution_budget,
    validate_topology,
)


def test_topology_accepts_three_pools_and_sparse_routes() -> None:
    pools, routes = validate_topology(
        {
            "execution": device(physical_capacity=2 << 30),
            "spill": pinned_host(capacity=1 << 30),
            "archive": pinned_host(capacity=2 << 30),
        },
        {
            "fetch": transfer_route(source="spill", destination="execution"),
            "evict": transfer_route(source="execution", destination="spill"),
            "archive_fetch": transfer_route(source="archive", destination="execution"),
        },
    )

    assert tuple(pools) == ("execution", "spill", "archive")
    assert tuple(routes) == ("fetch", "evict", "archive_fetch")


def test_topology_rejects_unknown_and_duplicate_route_endpoints() -> None:
    pools = {
        "execution": device(physical_capacity=2 << 30),
        "spill": pinned_host(capacity=1 << 30),
    }
    with pytest.raises(RuntimeConfigurationError, match="unknown pool"):
        validate_topology(
            pools,
            {"fetch": transfer_route(source="missing", destination="execution")},
        )
    with pytest.raises(RuntimeConfigurationError, match="must be unique"):
        validate_topology(
            pools,
            {
                "first": transfer_route(source="spill", destination="execution"),
                "second": transfer_route(source="spill", destination="execution"),
            },
        )


def test_topology_rejects_a_route_with_no_device_endpoint() -> None:
    with pytest.raises(RuntimeConfigurationError, match="exactly one endpoint"):
        validate_topology(
            {
                "execution": device(physical_capacity=2 << 30),
                "spill": pinned_host(capacity=1 << 30),
                "archive": pinned_host(capacity=1 << 30),
            },
            {
                "copy": transfer_route(source="spill", destination="archive"),
            },
        )


def test_execution_budget_accepts_runtime_physical_cap_without_double_charge() -> None:
    physical = 16 << 30
    derived = physical - (1280 << 20) - (256 << 20)
    pool = MemoryPool("execution", 0, "device", derived, physical, 0)

    assert resolve_execution_budget(None, pool) == derived
    assert resolve_execution_budget(physical, pool) == derived
    assert resolve_execution_budget(10 << 30, pool) == 10 << 30
    with pytest.raises(AdmissionError, match="falls between"):
        resolve_execution_budget(derived + 1, pool)
    with pytest.raises(AdmissionError, match="physical capacity"):
        resolve_execution_budget(physical + 1, pool)


def test_dynamic_scratch_reserve_is_an_optional_bounded_minimum() -> None:
    budget = 16 << 30
    assert resolve_dynamic_scratch_reserve(None, execution_budget=budget) == 0
    assert resolve_dynamic_scratch_reserve(1 << 30, execution_budget=budget) == (
        1 << 30
    )
    with pytest.raises(TypeError, match="integer byte count"):
        resolve_dynamic_scratch_reserve(True, execution_budget=budget)
    with pytest.raises(AdmissionError, match="non-negative"):
        resolve_dynamic_scratch_reserve(-1, execution_budget=budget)
    with pytest.raises(AdmissionError, match="exceeds"):
        resolve_dynamic_scratch_reserve(budget + 1, execution_budget=budget)


class _ThirdPartyPool(SpillPool):
    """A spill pool kind neither this module nor the runtime knows about.

    It stands in for a kind a loaded library registers, which is the point:
    nothing in the neutral configuration may need to have heard of a kind in
    order to carry it.
    """

    __slots__ = ()

    #: One object, not a fresh one per call -- the runtime borrows a pointer to
    #: it for the whole of bootstrap.
    _storage = ctypes.c_uint64(0xC0FFEE)

    @property
    def library(self) -> Path:
        return Path("/nowhere/libexample.so")

    def configuration(self) -> ctypes.Structure:
        return self._storage  # type: ignore[return-value]


def test_topology_carries_a_kind_it_has_never_heard_of() -> None:
    spill = _ThirdPartyPool(capacity=1 << 30, kind=7, kind_name="example")
    topology = Topology(
        *validate_topology(
            {
                "execution": device(physical_capacity=2 << 30),
                "spill": spill,
            },
            {
                "fetch": transfer_route(source="spill", destination="execution"),
                "evict": transfer_route(source="execution", destination="spill"),
            },
        )
    )

    bootstrap = topology.pool_bootstrap
    assert [item.kind for item in bootstrap] == [0, 7]
    assert bootstrap[1].capacity_bytes == 1 << 30
    assert bootstrap[1].library == Path("/nowhere/libexample.so")
    assert bootstrap[1].configuration is spill.configuration()
    # The device pool needs no kind configuration and names no library.
    assert bootstrap[0].configuration is None
    assert bootstrap[0].library is None

    pools = topology.pools(1 << 29)
    assert pools["spill"].kind == "example"
    assert pools["execution"].kind == "device"


def test_topology_admits_a_spill_pool_that_is_not_pinned_host() -> None:
    """A topology whose only spill pool is a registered kind is accepted.

    It used to be refused for want of a pinned-host pool specifically, which
    made the one built-in spill kind a requirement rather than a default.
    """

    pools, _ = validate_topology(
        {
            "execution": device(physical_capacity=2 << 30),
            "spill": _ThirdPartyPool(capacity=1 << 30, kind=7, kind_name="example"),
        },
        {"fetch": transfer_route(source="spill", destination="execution")},
    )
    assert not any(isinstance(value, PinnedHostPool) for value in pools.values())
