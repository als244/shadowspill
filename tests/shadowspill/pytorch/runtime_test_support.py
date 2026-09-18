"""Shared process-lifetime Runtime for in-process PyTorch public-API tests."""

from shadowspill.memory import device, transfer_route
from shadowspill.pytorch import Runtime

from ...spill_pool import spill_pool

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
                "spill": spill_pool(1 << 30),
            },
            routes={
                "fetch": transfer_route(source="spill", destination="execution"),
                "evict": transfer_route(source="execution", destination="spill"),
            },
        )
    return _RUNTIME
