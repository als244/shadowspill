"""Shared process-lifetime Runtime for in-process PyTorch public-API tests."""

import pytest
import torch

from shadowspill.memory import device, transfer_route
from shadowspill.pytorch import Runtime
from shadowspill.runtime.configuration import adapter_path

from ...spill_pool import spill_pool

_RUNTIME: Runtime | None = None


def require_adapter() -> None:
    if torch.cuda.is_initialized():
        pytest.skip("public allocator installation requires a fresh process")
    try:
        adapter_path(None)
    except RuntimeError:
        pytest.skip("the built PyTorch adapter is not installed")


def public_test_runtime() -> Runtime:
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = Runtime(
            pools={
                # Room for the largest test step on any device: a plan keeps
                # back its largest matmul workspace, which differs by
                # architecture -- tens of MiB on some, a few on others.
                "execution": device(
                    physical_capacity=3 << 30,
                    external_headroom=512 << 20,
                ),
                "spill": spill_pool(1 << 30),
            },
            routes={
                "fetch": transfer_route(source="spill", destination="execution"),
                "evict": transfer_route(source="execution", destination="spill"),
            },
        )
    return _RUNTIME
