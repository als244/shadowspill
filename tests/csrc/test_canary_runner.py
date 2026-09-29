"""The suite wrapper streams progress and cannot leave timed-out children running."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.csrc.test_canaries import _stream_process


def test_process_output_reaches_console_and_failure_evidence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = _stream_process(
        [
            sys.executable,
            "-c",
            "print('canary progress', flush=True); raise SystemExit(3)",
        ],
        cwd=tmp_path,
        timeout=5,
    )
    assert result.returncode == 3
    assert result.stdout == "canary progress\n"
    assert capsys.readouterr().out == result.stdout


def test_deadline_still_fires_when_a_child_keeps_stdout_open(tmp_path: Path) -> None:
    # The launcher exits immediately, but its sleeping child inherits the
    # pipe. Waiting for EOF without a deadline would hang for a full minute.
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        _stream_process(
            [
                sys.executable,
                "-c",
                "import subprocess, sys; "
                "subprocess.Popen([sys.executable, '-c', "
                "'import time; time.sleep(60)'])",
            ],
            cwd=tmp_path,
            timeout=0.5,
        )
    assert time.monotonic() - started < 5
