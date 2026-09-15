"""What a caller asks of the runtime, checked: pools and routes, budgets, the device."""

from __future__ import annotations

import pytest

from shadowspill.errors import AdmissionError
from shadowspill.memory import device, pinned_host, transfer_route
from shadowspill.runtime import MemoryPool
from shadowspill.runtime.configuration import (
    RuntimeConfigurationError,
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


def test_topology_rejects_routes_without_a_supported_backend_pair() -> None:
    with pytest.raises(RuntimeConfigurationError, match="only between"):
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
