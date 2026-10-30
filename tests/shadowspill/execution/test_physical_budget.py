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


def test_a_worker_reports_overruns_unless_told_to_reject(monkeypatch):
    import torch

    from shadowspill.execution import setup
    from shadowspill.memory import PinnedHostPool

    seen = []

    class Runtime:
        def __init__(self, *, frontend, pools, routes, calibrate):
            seen.append(pools["execution"])

        def close(self):
            pass

    identity = {"torch": torch.__version__}
    monkeypatch.setattr(setup, "Runtime", Runtime)
    monkeypatch.setattr(setup, "PyTorchFrontend", lambda: None)
    monkeypatch.setattr(setup, "require_compiled_device", lambda program, device: None)
    monkeypatch.setattr(setup, "build_setup", lambda device: identity)
    plan = SimpleNamespace(
        simulation_only_reasons=(),
        program=SimpleNamespace(devices=("cuda_0",), setup=identity),
        memory_budgets=SimpleNamespace(execution_bytes=8 << 30, spill_bytes=1 << 30),
        communication_info=SimpleNamespace(metadata={}),
    )
    setup.open_worker(plan, 0, 512 << 20, PinnedHostPool(1 << 30))
    setup.open_worker(plan, 0, 512 << 20, PinnedHostPool(1 << 30), True)
    assert [pool.reject_overbudget for pool in seen] == [False, True]
    assert all(pool.external_headroom == 512 << 20 for pool in seen)
