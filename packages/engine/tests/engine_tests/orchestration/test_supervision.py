"""Real owned children, bounded log pressure and scoped process-group cleanup."""

import os
import signal
import subprocess
import sys
import time
from threading import Event, Thread

import pytest

from shadowspill.orchestration.launch.supervision import OwnedProcess


def start(root, code, *arguments, limit=4096):
    return OwnedProcess(
        [sys.executable, "-u", "-c", code, *map(str, arguments)],
        environment={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        directory=root / "logs",
        cwd=root,
        log_limit_bytes=limit,
    )


def await_file(path):
    deadline = time.monotonic() + 5
    while not path.exists():
        assert time.monotonic() < deadline
        time.sleep(0.005)


def test_large_logs_are_drained_with_a_fixed_file_limit_and_no_child_deadlock(tmp_path):
    count = (1 << 20) + 17
    code = f"import os; os.write(1, b'o' * {count}); os.write(2, b'e' * {count})"
    with start(tmp_path, code) as child:
        assert child.wait(timeout=5) == 0
        # Observation must not reap the leader: its PID still protects group
        # ownership even if the node daemon delays its final descendant cleanup.
        status = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOWAIT)
        assert status.si_status == 0
    assert child.returncode == 0 and child.closed
    for name, status in zip(("stdout", "stderr"), child.logs, strict=True):
        assert status.bytes_seen == count and status.bytes_written == 4096
        assert status.truncated and status.error is None
        assert (tmp_path / "logs" / f"{name}.log").stat().st_size == 4096
    with pytest.raises(ChildProcessError):
        os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG)
    child.close()  # Cannot signal a PID again after it has been reaped.


def test_simultaneous_shutdown_owners_cannot_reap_or_signal_the_group_twice(
    tmp_path, monkeypatch
):
    child = start(tmp_path, "import time; time.sleep(60)")
    entered, release, second_wait = Event(), Event(), Event()
    original = child._process.wait
    calls, errors = [], []

    def wait(*args, **kwargs):
        calls.append(True)
        if len(calls) > 1:
            second_wait.set()
        entered.set()
        assert release.wait(3)
        return original(*args, **kwargs)

    monkeypatch.setattr(child._process, "wait", wait)

    def close():
        try:
            child.close(grace_seconds=0, terminate_seconds=0)
        except BaseException as error:
            errors.append(error)

    first, second = Thread(target=close), Thread(target=close)
    first.start()
    assert entered.wait(3)
    second.start()
    try:
        assert not second_wait.wait(0.05)
    finally:
        release.set()
        first.join(timeout=3)
        second.join(timeout=3)
        child.close(grace_seconds=0, terminate_seconds=0)
    assert not first.is_alive() and not second.is_alive()
    assert not errors and calls == [True] and child.closed


def test_explicit_close_escalates_only_the_owned_group_and_has_no_idle_timeout(
    tmp_path,
):
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    right.mkdir()
    code = """
import signal, sys, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path(sys.argv[1]).write_text('ready')
while True:
    time.sleep(1)
"""
    first = start(left, code, left / "ready")
    second = start(right, code, right / "ready")
    try:
        await_file(left / "ready")
        await_file(right / "ready")
        with pytest.raises(subprocess.TimeoutExpired):
            first.wait(timeout=0.03)
        assert first.returncode is None and second.returncode is None
        first.close(grace_seconds=0, terminate_seconds=0.03)
        assert first.returncode == -signal.SIGKILL
        assert second.returncode is None
    finally:
        first.close(grace_seconds=0, terminate_seconds=0)
        second.close(grace_seconds=0, terminate_seconds=0)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process-state observation")
def test_descendants_cannot_keep_log_pipes_or_processes_after_leader_exit(tmp_path):
    code = """
import subprocess, sys
from pathlib import Path
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
Path(sys.argv[1]).write_text(str(child.pid))
"""
    with start(tmp_path, code, tmp_path / "descendant") as child:
        assert child.wait(timeout=5) == 0
        descendant = int((tmp_path / "descendant").read_text())
    deadline = time.monotonic() + 5
    state_path = type(tmp_path)(f"/proc/{descendant}/stat")
    while True:
        try:
            state = state_path.read_text()
        except (FileNotFoundError, ProcessLookupError):
            # procfs can open the inode before its task disappears during read.
            break
        if state.rsplit(")", 1)[1].strip().split()[0] == "Z":
            break
        assert time.monotonic() < deadline, "owned descendant survived group cleanup"
        time.sleep(0.005)
    assert child.closed and all(not log.thread.is_alive() for log in child._logs)


def test_log_write_failure_is_reported_but_remaining_bytes_are_still_drained(tmp_path):
    code = """
import os, sys, time
from pathlib import Path
while not Path(sys.argv[1]).exists():
    time.sleep(0.005)
os.write(1, b'x' * (1 << 20))
"""
    with start(tmp_path, code, tmp_path / "go") as child:
        child._logs[0].destination.close()
        (tmp_path / "go").touch()
        assert child.wait(timeout=5) == 0
    assert child.logs[0].bytes_seen == 1 << 20
    assert child.logs[0].error and child.logs[0].truncated


def test_failed_launch_closes_logs_and_never_overwrites_an_existing_directory(tmp_path):
    options = dict(environment=os.environ, directory=tmp_path / "logs", cwd=tmp_path)
    with pytest.raises(FileNotFoundError):
        OwnedProcess(["/no/such/worker"], **options)
    assert sorted(path.name for path in (tmp_path / "logs").iterdir()) == [
        "stderr.log",
        "stdout.log",
    ]
    with pytest.raises(FileExistsError):
        OwnedProcess([sys.executable, "-c", "pass"], **options)
