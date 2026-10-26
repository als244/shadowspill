"""Physical checks preserve admission and report the actual reserved quantities."""

import ctypes
from types import SimpleNamespace

import pytest

from shadowspill.errors import AdmissionError
from shadowspill.execution.distributed.checks import check_external_budget
from shadowspill.runtime.abi import PhysicalMemory


@pytest.mark.parametrize("available", [True, False])
def test_overbudget_check_reports_memory_without_masking_failure(available):
    def physical(pointer):
        value = ctypes.cast(pointer, ctypes.POINTER(PhysicalMemory)).contents
        value.process_bytes = 1200
        return 0 if available else 1

    runtime = SimpleNamespace(
        _installed=SimpleNamespace(
            library=SimpleNamespace(
                shadowspill_pytorch_check_physical_budget=lambda: 43,
                shadowspill_pytorch_physical_memory=physical,
            ),
            admission=SimpleNamespace(
                baseline_bytes=100,
                allocator_pool_bytes=600,
                device_budget_bytes=1000,
                external_headroom_bytes=300,
            ),
        )
    )
    with pytest.raises(AdmissionError, match="status 43") as failure:
        check_external_budget(runtime)
    if available:
        message = str(failure.value)
        assert "process=1200" in message and "budget=1000" in message
        assert "external=500" in message and "external_headroom=300" in message


def test_within_budget_does_not_query_extra_memory():
    runtime = SimpleNamespace(
        _installed=SimpleNamespace(
            library=SimpleNamespace(shadowspill_pytorch_check_physical_budget=lambda: 0)
        )
    )
    check_external_budget(runtime)
