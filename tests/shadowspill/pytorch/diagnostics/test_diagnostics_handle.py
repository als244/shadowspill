"""A traced call's diagnostics are collected once, whether or not that works."""

from __future__ import annotations

import pytest

from shadowspill.pytorch.diagnostics.step import DiagnosticsHandle


def test_diagnostics_are_collected_once() -> None:
    calls: list[int] = []
    evidence = object()

    def collect() -> object:
        calls.append(1)
        return evidence

    handle = DiagnosticsHandle(collect)  # type: ignore[arg-type]
    assert not handle.resolved
    assert handle.result() is evidence
    assert handle.result() is evidence
    assert handle.resolved
    assert calls == [1]


def test_a_failed_collection_keeps_its_error_and_is_not_run_again() -> None:
    """The first attempt consumes the measurement, so a second could only
    report that nothing is armed; the original error is raised instead, and
    the handle counts as resolved, so closing does not collect again."""

    calls: list[int] = []

    def collect() -> object:
        calls.append(1)
        raise RuntimeError("transfer count mismatch")

    handle = DiagnosticsHandle(collect)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="transfer count mismatch"):
        handle.result()
    assert handle.resolved
    with pytest.raises(RuntimeError, match="transfer count mismatch"):
        handle.result()
    assert calls == [1]
