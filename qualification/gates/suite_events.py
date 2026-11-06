"""Flush pytest results as they happen, including the active test at a crash."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--suite-events",
        type=Path,
        help="append incremental qualification results to this JSON-lines file",
    )


def pytest_configure(config: pytest.Config) -> None:
    path = config.getoption("suite_events")
    # A distributed pytest controller receives the workers' reports. Workers
    # must not open the same file independently.
    if path is None or hasattr(config, "workerinput"):
        return
    events = SuiteEvents(path)
    config.pluginmanager.register(events, "qualification-suite-events")
    config.add_cleanup(events.close)


class SuiteEvents:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.session = uuid.uuid4().hex
        self.handle = self.path.open("a", encoding="utf-8")
        # Separate a new session even if the previous process died mid-record.
        self.handle.write("\n")

    def close(self) -> None:
        self.handle.close()

    def write(self, event: str, **values: Any) -> None:
        record = {
            "schema": 1,
            "session": self.session,
            "event": event,
            "time_ns": time.time_ns(),
            **values,
        }
        self.handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        # Preserve completed records after process failure without a per-test
        # disk synchronization. This is not protection against power loss.
        self.handle.flush()

    def pytest_sessionstart(self, session: pytest.Session) -> None:
        self.write("session_start", pid=os.getpid())
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_line(
                f"suite events session={self.session} path={self.path}"
            )

    def pytest_deselected(self, items: list[pytest.Item]) -> None:
        self.write("deselected", nodeids=[item.nodeid for item in items])

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        self.write("collected", nodeids=[item.nodeid for item in session.items])

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        if not report.passed:
            self.write(
                "collection_result",
                nodeid=report.nodeid,
                outcome=report.outcome,
                detail=report.longreprtext,
            )

    def pytest_runtest_logstart(self, nodeid: str) -> None:
        self.write("test_start", nodeid=nodeid)

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        values: dict[str, Any] = {
            "nodeid": report.nodeid,
            "when": report.when,
            "outcome": report.outcome,
            "duration_seconds": report.duration,
        }
        if not report.passed:
            values["detail"] = report.longreprtext
        if hasattr(report, "wasxfail"):
            values["wasxfail"] = report.wasxfail
        self.write("test_result", **values)

    def pytest_runtest_logfinish(self, nodeid: str) -> None:
        self.write("test_finish", nodeid=nodeid)

    def pytest_sessionfinish(self, exitstatus: int) -> None:
        self.write("session_finish", exitstatus=int(exitstatus))
