from __future__ import annotations

import pytest

from shadowspill.memory import device, pinned_host, transfer_route
from shadowspill.pytorch.runtime_adapter.runtime import (
    RuntimeConfigurationError,
    _validate_topology,
)


def test_topology_accepts_three_pools_and_sparse_routes() -> None:
    pools, routes = _validate_topology(
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
        _validate_topology(
            pools,
            {"fetch": transfer_route(source="missing", destination="execution")},
        )
    with pytest.raises(RuntimeConfigurationError, match="must be unique"):
        _validate_topology(
            pools,
            {
                "first": transfer_route(source="spill", destination="execution"),
                "second": transfer_route(source="spill", destination="execution"),
            },
        )


def test_topology_rejects_routes_without_a_supported_backend_pair() -> None:
    with pytest.raises(RuntimeConfigurationError, match="only between"):
        _validate_topology(
            {
                "execution": device(physical_capacity=2 << 30),
                "spill": pinned_host(capacity=1 << 30),
                "archive": pinned_host(capacity=1 << 30),
            },
            {
                "copy": transfer_route(source="spill", destination="archive"),
            },
        )
