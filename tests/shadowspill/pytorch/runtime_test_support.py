"""Shared process-lifetime Runtime for in-process PyTorch public-API tests."""

from shadowspill.memory import device, pinned_host, transfer_route
from shadowspill.pytorch import Runtime

_RUNTIME: Runtime | None = None


def public_test_runtime() -> Runtime:
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = Runtime(
            pools={
                "execution": device(
                    physical_capacity=2 << 30,
                    provider_headroom=512 << 20,
                ),
                "spill": pinned_host(capacity=1 << 30),
            },
            routes={
                "fetch": transfer_route(source="spill", destination="execution"),
                "evict": transfer_route(source="execution", destination="spill"),
            },
        )
    return _RUNTIME
