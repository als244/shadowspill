"""A runtime close is refused, or it releases everything and reports what it saw.

The adapter close returns INVALID_STATE both when it refuses, because
caller-owned outputs remain, and when it released everything after that same
status was latched. Only a refused close leaves the runtime published.
Closing frees the neutral runtime, so every later call is refused before it
reaches the library.
"""

import threading
from types import SimpleNamespace

import pytest

from shadowspill.runtime import Runtime, RuntimeConfigurationError, core
from shadowspill.status import Status


class _Adapter:
    def __init__(self, *, refuse: bool) -> None:
        self.refuse = refuse
        self.published = True

    def shadowspill_pytorch_allocator_close(self) -> int:
        if not self.refuse:
            self.published = False
        return Status.INVALID_STATE

    def shadowspill_pytorch_runtime_handle(self, pointer: object) -> int:
        return Status.OK if self.published else Status.CLOSED


def _runtime(adapter: _Adapter) -> Runtime:
    runtime = Runtime.__new__(Runtime)
    runtime._lock = threading.RLock()
    runtime._closed = False
    runtime._active_plan_handles = set()
    runtime._pending_plan_handle = None
    runtime._active_object_references = 0
    runtime._installed = SimpleNamespace(library=adapter)
    return runtime


def test_a_refused_close_leaves_the_runtime_open() -> None:
    runtime = _runtime(_Adapter(refuse=True))

    with pytest.raises(RuntimeConfigurationError, match="caller-owned device outputs"):
        runtime.close()

    assert not runtime._closed


def test_a_close_that_observed_a_latched_invalid_state_released_the_runtime() -> None:
    runtime = _runtime(_Adapter(refuse=False))

    with pytest.raises(RuntimeConfigurationError, match="observing status 42"):
        runtime.close()

    assert runtime._closed
    runtime.close()


class _ClosingAdapter:
    def shadowspill_pytorch_allocator_close(self) -> int:
        return Status.OK


def _unreachable() -> object:
    raise AssertionError("a closed runtime reached the library")


def test_a_closed_runtime_refuses_every_call_before_the_library(monkeypatch) -> None:
    runtime = _runtime(_ClosingAdapter())
    runtime._unusable_reason = None
    runtime._runtime_handle = 1
    runtime._pools = {"execution": SimpleNamespace(pool_id=0)}
    runtime.close()
    monkeypatch.setattr(core, "runtime_library", _unreachable)

    for call in (
        runtime.pool_statistics,
        runtime.next_plan_id,
        lambda: runtime.plan_state(1),
        lambda: runtime.transfer_capabilities,
        runtime.calibrate_transfer_capabilities,
        runtime.__enter__,
    ):
        with pytest.raises(RuntimeConfigurationError, match="Runtime is closed"):
            call()
    assert runtime._runtime_handle == 0
