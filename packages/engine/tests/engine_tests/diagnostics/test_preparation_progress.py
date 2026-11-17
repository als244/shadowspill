"""Host log formatting preserves computation, failures and nested scope."""

import re

import pytest

from shadowspill.diagnostics.progress import Progress, event, phase, timed


def test_nested_operations_have_timestamps_hierarchy_and_durations():
    lines = []
    reporter = Progress(lines.append)
    with reporter.activate(), phase("Build"):
        with phase("Candidate 1/3"):
            event("Workers: 1")
        with pytest.raises(ValueError), phase("Compile"):
            raise ValueError("actual failure")
    assert len(lines) == 7  # Actual boundaries only, no background events.
    messages = [line.split("] ", 1)[1] for line in lines]
    assert messages[:3] == ["→ Build", "  → Candidate 1/3", "    Workers: 1"]
    assert messages[3].startswith("  ✓ Candidate 1/3 — ")
    assert messages[4] == "  → Compile"
    assert messages[5].startswith("  ✗ ValueError: Compile — ")
    assert messages[6].startswith("✓ Build — ")
    assert all(
        re.match(
            r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z \[rank=\d+ pid=\d+ \+\d+\.\d\ds\]",
            line,
        )
        for line in lines
    )


def test_disabled_or_broken_log_sink_preserves_result_and_original_failure():
    calls = []

    @timed("Computation")
    def compute(value):
        calls.append(value)
        if value < 0:
            raise ValueError("original")
        return value + 1

    assert compute(1) == 2

    def broken(message):
        raise BrokenPipeError("display closed")

    with Progress(broken).activate():
        assert compute(2) == 3
        with pytest.raises(ValueError, match="original"):
            compute(-1)
    assert calls == [1, 2, -1]


def test_nested_reporter_restores_outer_scope():
    outer, inner = [], []
    with Progress(outer.append).activate(), phase("Outer"):
        with Progress(inner.append).activate(), phase("Inner"):
            pass
        event("Back")
    assert len(outer) == 3 and len(inner) == 2
    assert outer[1].endswith("  Back")


def test_recoverably_rejected_candidate_is_not_logged_as_successful():
    lines = []
    with Progress(lines.append).activate():
        with phase("Candidate 1/3") as outcome:
            outcome.rejected = True
        with phase("Candidate 2/3"):
            pass
    assert "⊘ Rejected: Candidate 1/3" in lines[1]
    assert "✓ Candidate 2/3" in lines[3]
    with phase("Silent") as outcome:
        outcome.rejected = True
