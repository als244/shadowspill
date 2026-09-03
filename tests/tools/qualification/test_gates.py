"""Running the gates in one command."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from tools.qualification.gates import GATE_ORDER, _suite_report, run_gates


class _FakeProcess:
    """Enough of Popen for a gate: lines to stream, then a return code."""

    def __init__(self, returncode: int) -> None:
        self._returncode = returncode
        self.stdout = iter(["one line of gate output\n"])

    def wait(self) -> int:
        return self._returncode


def _record(
    monkeypatch: pytest.MonkeyPatch, failures: set[str] | None = None
) -> list[tuple[str, ...]]:
    """Capture the commands run instead of running them."""

    calls: list[tuple[str, ...]] = []
    refused = failures or set()

    def fake_popen(command: list[str], **kwargs: Any) -> _FakeProcess:
        calls.append(tuple(command))
        named = next(
            (gate for gate in GATE_ORDER if gate in " ".join(command)), "suite"
        )
        return _FakeProcess(1 if named in refused else 0)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    return calls


def _gate_of(command: tuple[str, ...]) -> str:
    joined = " ".join(command)
    if "pytest" in joined:
        return "suite"
    return "numerical" if "numerical" in joined else "performance"


def test_gates_run_in_a_fixed_order_whatever_order_they_are_asked_for(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    calls = _record(monkeypatch)

    outcomes = run_gates(["performance", "suite", "numerical"], run="demo")

    assert [outcome.name for outcome in outcomes] == list(GATE_ORDER)
    assert [_gate_of(call) for call in calls] == list(GATE_ORDER)
    assert all(outcome.passed for outcome in outcomes)


def test_a_subset_runs_only_what_was_asked_for(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    calls = _record(monkeypatch)

    outcomes = run_gates(["numerical"], run="demo")

    assert [outcome.name for outcome in outcomes] == ["numerical"]
    assert [_gate_of(call) for call in calls] == ["numerical"]


def test_each_gate_writes_its_output_under_the_run_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    calls = _record(monkeypatch)

    outcomes = run_gates(list(GATE_ORDER), run="0903")

    joined = [" ".join(call) for call in calls]
    assert "qualification/results/numerical_0903" in joined[1]
    assert "qualification/results/performance_0903" in joined[2]
    assert all(outcome.log.parent.name == "gates_0903" for outcome in outcomes)
    assert {outcome.log.name for outcome in outcomes} == {
        f"{gate}.log" for gate in GATE_ORDER
    }


def test_a_failure_stops_the_gates_that_would_follow(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    _record(monkeypatch, failures={"numerical"})

    outcomes = run_gates(list(GATE_ORDER), run="demo")

    assert [outcome.name for outcome in outcomes] == ["suite", "numerical"]
    assert not outcomes[-1].passed


def test_continuing_after_a_failure_runs_the_rest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    _record(monkeypatch, failures={"suite"})

    outcomes = run_gates(list(GATE_ORDER), run="demo", continue_after_failure=True)

    assert [outcome.name for outcome in outcomes] == list(GATE_ORDER)
    assert [outcome.passed for outcome in outcomes] == [False, True, True]


def test_keep_going_is_forwarded_only_to_the_matrices(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    calls = _record(monkeypatch)

    run_gates(list(GATE_ORDER), run="demo", keep_going=True)

    assert "--keep-going" not in calls[0]
    assert "--keep-going" in calls[1]
    assert "--keep-going" in calls[2]


def test_the_summary_names_the_tests_that_failed(tmp_path: Path) -> None:
    log = tmp_path / "suite.log"
    log.write_text(
        "=========================== short test summary info ====================\n"
        "FAILED tests/shadowspill/planner/test_pressurefit.py::test_a_thing - err\n"
        "ERROR tests/shadowspill/simulator/test_failures.py::test_another\n"
        "2 failed, 810 passed, 1 skipped, 5 deselected in 143.82s\n"
    )

    rows = _suite_report(log)

    assert rows[0] == "    2 failed, 810 passed, 1 skipped"
    assert any("test_a_thing" in row for row in rows)
    assert any("test_another" in row for row in rows)


def test_a_long_failure_list_points_at_the_log_instead(tmp_path: Path) -> None:
    log = tmp_path / "suite.log"
    failures = "\n".join(f"FAILED tests/x.py::test_{index}" for index in range(40))
    log.write_text(f"{failures}\n40 failed, 1 passed in 10.00s\n")

    rows = _suite_report(log)

    assert sum("FAILED" in row for row in rows) == 15
    assert any("and 25 more" in row and "suite.log" in row for row in rows)


def test_a_clean_suite_names_nothing(tmp_path: Path) -> None:
    log = tmp_path / "suite.log"
    log.write_text("812 passed in 143.82s\n")

    rows = _suite_report(log)

    assert rows == ["    812 passed"]


def test_tests_run_under_ctest_are_not_reported_as_excluded(tmp_path: Path) -> None:
    log = tmp_path / "suite.log"
    log.write_text("812 passed, 5 deselected in 143.82s\n")

    rows = _suite_report(log)

    assert rows == ["    812 passed"]


def test_a_run_is_named_for_the_commit_it_measured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.qualification import gates

    def fake_git(command: list[str], **kwargs: Any) -> Any:
        class _Result:
            stdout = "abc1234\n" if "rev-parse" in command else ""

        return _Result()

    monkeypatch.setattr(subprocess, "run", fake_git)

    assert gates._default_run().startswith("abc1234_")
    assert "dirty" not in gates._default_run()


def test_a_modified_tree_says_so_in_the_run_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.qualification import gates

    def fake_git(command: list[str], **kwargs: Any) -> Any:
        class _Result:
            stdout = "abc1234\n" if "rev-parse" in command else " M src/thing.py\n"

        return _Result()

    monkeypatch.setattr(subprocess, "run", fake_git)

    assert gates._default_run().startswith("abc1234_dirty_")
