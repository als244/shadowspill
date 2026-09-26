"""A fake runtime for the planned callables, and the plan-lifecycle functions
routed to it.

The callables call functions over a runtime rather than methods on it, so a test
replaces the functions in the callables module; each checks the plan handle the
callable was given and records on the fake runtime what it was asked to do.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import shadowspill.pytorch.callables as callable_module


class FakeRuntime:
    def __init__(self, *, fail_release: bool = False) -> None:
        self.fail_release = fail_release
        self.adopted = False
        self.prepared_error: BaseException | None = None
        self.released = False
        self.residue_reclaimed = False
        # No plan shares another's slab here.
        self._installed = SimpleNamespace(slab_hosts={})


def fake_plan_lifecycle(monkeypatch: pytest.MonkeyPatch, *, plan_handle: int) -> None:
    def adopt(runtime: FakeRuntime, handle: int) -> None:
        assert handle == plan_handle
        runtime.adopted = True

    def prepare(runtime: FakeRuntime, error: BaseException, **kwargs: object) -> None:
        del kwargs
        runtime.prepared_error = error

    def release(runtime: FakeRuntime, handle: int) -> None:
        assert handle == plan_handle
        runtime.released = True
        if runtime.fail_release:
            raise RuntimeError("plan cleanup failed")

    def reclaim(runtime: FakeRuntime, handle: int) -> None:
        assert handle == plan_handle
        runtime.residue_reclaimed = True

    def wait(handle: int) -> None:
        assert handle == plan_handle

    monkeypatch.setattr(callable_module, "adopt_plan", adopt)
    monkeypatch.setattr(callable_module, "prepare_failure_cleanup", prepare)
    monkeypatch.setattr(callable_module, "release_plan", release)
    monkeypatch.setattr(callable_module, "reclaim_plan_scoped_residue", reclaim)
    monkeypatch.setattr(callable_module, "wait_plan_idle", wait)
