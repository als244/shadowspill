"""A crashing pytest subprocess must leave its preceding failures reviewable."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _run(directory: Path, source: str, *options: str) -> subprocess.CompletedProcess:
    test = directory / "test_case.py"
    test.write_text(source)
    configuration = directory / "pytest.ini"
    configuration.write_text("[pytest]\n")
    root = Path(__file__).resolve().parents[2]
    environment = {
        **os.environ,
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join((str(root), os.environ.get("PYTHONPATH", ""))),
    }
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(configuration),
            "--rootdir",
            str(directory),
            "--confcutdir",
            str(directory),
            "-q",
            "-p",
            "no:cacheprovider",
            "-p",
            "qualification.gates.suite_events",
            "--suite-events",
            str(directory / "events.jsonl"),
            str(test),
            *options,
        ],
        cwd=directory,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _records(directory: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (directory / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]


def test_failures_and_active_test_survive_abrupt_exit(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        """import os
def test_first():
    assert False, "preserved failure before process exit"
def test_abrupt():
    os._exit(23)
""",
    )
    assert result.returncode == 23, result.stdout + result.stderr
    records = _records(tmp_path)
    assert len({record["session"] for record in records}) == 1
    assert records[0]["event"] == "session_start"
    assert not any(record["event"] == "session_finish" for record in records)
    failure = next(record for record in records if record.get("outcome") == "failed")
    assert failure["when"] == "call"
    assert "preserved failure before process exit" in failure["detail"]
    assert records[-1]["nodeid"].endswith("::test_abrupt")
    assert records[-1]["event"] == "test_result" and records[-1]["when"] == "setup"
    assert not any(
        record["event"] == "test_finish" and record["nodeid"].endswith("::test_abrupt")
        for record in records
    )
    # A later session appends its own identity; it cannot finish the failed one.
    second = _run(tmp_path, "def test_second(): pass\n")
    assert second.returncode == 0, second.stdout + second.stderr
    appended = _records(tmp_path)
    assert appended[: len(records)] == records
    assert len({record["session"] for record in appended}) == 2
    assert appended[-1]["event"] == "session_finish"
    assert appended[-1]["exitstatus"] == 0
    assert appended[-1]["session"] != records[0]["session"]


def test_setup_failure_skip_and_deselection_are_distinct(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        """import pytest
@pytest.fixture
def broken():
    raise RuntimeError("setup evidence")
def test_setup(broken): pass
def test_skipped(): pytest.skip("not supported")
def test_omitted(): assert False
""",
        "-k",
        "not omitted",
    )
    assert result.returncode == 1, result.stdout + result.stderr
    records = _records(tmp_path)
    selected = next(
        record["nodeids"] for record in records if record["event"] == "collected"
    )
    omitted = next(
        record["nodeids"] for record in records if record["event"] == "deselected"
    )
    assert len(selected) == 2 and len(omitted) == 1
    assert omitted[0].endswith("::test_omitted")
    failure = next(record for record in records if record.get("outcome") == "failed")
    assert failure["when"] == "setup" and "setup evidence" in failure["detail"]
    skipped = next(record for record in records if record.get("outcome") == "skipped")
    assert "not supported" in skipped["detail"]
    assert records[-1]["event"] == "session_finish"
    assert records[-1]["exitstatus"] == 1


def test_collection_failure_is_recorded(tmp_path: Path) -> None:
    result = _run(tmp_path, 'raise RuntimeError("collection evidence")\n')
    assert result.returncode == 2, result.stdout + result.stderr
    records = _records(tmp_path)
    failure = next(
        record for record in records if record["event"] == "collection_result"
    )
    assert failure["outcome"] == "failed" and "collection evidence" in failure["detail"]
    assert records[-1]["event"] == "session_finish"
    assert records[-1]["exitstatus"] == 2
