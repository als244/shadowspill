"""Optional trace assembly cannot turn a confirmed update into unknown execution."""

from types import SimpleNamespace

import pytest

from shadowspill.diagnostics import collection
from shadowspill.diagnostics.collection import DiagnosticAssemblyError
from shadowspill.execution.tracing import InvocationTrace
from shadowspill.pytorch.invocation import InvocationResult


def test_only_post_drain_assembly_failure_is_optional(monkeypatch):
    calls = []
    monkeypatch.setattr(
        collection, "validate_completed_timing", lambda _: calls.append("validate")
    )
    monkeypatch.setattr(
        collection, "resolve_trace_evidence", lambda *_: calls.append("drain")
    )

    def failed(*_):
        calls.append("assemble")
        raise ValueError("bad view" + "x" * 2000)

    monkeypatch.setattr(collection, "_assemble", failed)
    with pytest.raises(DiagnosticAssemblyError) as raised:
        collection.collect_step_diagnostics(None, None)
    assert calls == ["validate", "drain", "assemble"]
    assert len(str(raised.value)) == 1024
    assert raised.value.__context__ is None

    def incomplete(*_):
        raise RuntimeError("terminal transfer failed")

    monkeypatch.setattr(collection, "resolve_trace_evidence", incomplete)
    with pytest.raises(RuntimeError, match="terminal transfer") as raised:
        collection.collect_step_diagnostics(None, None)
    assert not isinstance(raised.value, DiagnosticAssemblyError)


@pytest.mark.parametrize("fault", [None, "assembly", "completion", "cleanup"])
def test_invocation_completion_and_optional_view_have_distinct_error_boundaries(fault):
    events, failures = [], []
    trace = object.__new__(InvocationTrace)
    trace.result = trace._error = None
    evidence = object()

    def complete():
        events.append("complete")
        if fault == "completion":
            raise RuntimeError("completion failed")

    def collect():
        events.append("collect")
        if fault == "assembly":
            raise DiagnosticAssemblyError("view failed")
        return evidence

    def release():
        events.append("release")
        if fault == "cleanup":
            raise RuntimeError("cleanup failed")

    trace.timing = SimpleNamespace(
        mark_cycle_end=complete, collect_step_diagnostics=collect, release=release
    )
    handle = InvocationResult(
        "trained",
        trace.finish,
        on_resolved=lambda _: events.append("resolved"),
        on_failure=failures.append,
        diagnostics=lambda: trace.diagnostics,
    )
    if fault in ("completion", "cleanup"):
        with pytest.raises(RuntimeError, match=fault):
            handle.result()
        assert len(failures) == 1
    else:
        assert handle.result() == "trained"
        assert not failures
        if fault == "assembly":
            with pytest.raises(DiagnosticAssemblyError, match="view failed"):
                _ = handle.diagnostics
            assert handle.result() == "trained"
        else:
            assert handle.diagnostics is evidence
    assert events[-2:] == ["release", "resolved"]
    assert events.count("complete") == 1
