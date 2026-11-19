"""Kill a real node daemon process and confirm even detached descendants are reaped."""

import fcntl
import json
import os
import signal
import subprocess
import sys
from contextlib import suppress
from time import monotonic, sleep

import pytest

from shadowspill.orchestration.launch.supervision import OwnedProcess

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="Linux node daemon supervisor"
)


def wait_file(path, process=None):
    deadline = monotonic() + 5
    while not path.exists():
        if process is not None:
            assert process.poll() is None
        assert monotonic() < deadline
        sleep(0.02)
    return path.read_text()


def test_guarded_worker_preserves_return_code_and_bounded_output(tmp_path):
    process = OwnedProcess(
        [sys.executable, "-c", "print('worker output'); raise SystemExit(7)"],
        environment=os.environ,
        directory=tmp_path / "logs",
        cwd=tmp_path,
        guard_fds=(),
    )
    assert process.wait(timeout=5) == 7
    process.close()
    assert (tmp_path / "logs/stdout.log").read_text() == "worker output\n"
    assert process.closed


def test_node_daemon_sigkill_reaps_session_escaped_descendant_before_unlock(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text("""
import json, os, signal, time
from pathlib import Path
child = os.fork()
if child == 0:
    os.setsid()
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    Path('detached').write_text(str(os.getpid()))
    while True:
        time.sleep(1)
Path('worker-pids').write_text(json.dumps([os.getpid(), child]))
while True:
    time.sleep(1)
""")
    daemon = tmp_path / "daemon.py"
    daemon.write_text("""
import fcntl, os, sys, time
from pathlib import Path
from shadowspill.orchestration.launch.supervision import OwnedProcess
lock = open('ownership.lock', 'w')
fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
process = OwnedProcess(
    [sys.executable, 'worker.py'], environment=os.environ,
    directory=Path('logs'), cwd=Path.cwd(), guard_fds=(lock.fileno(),),
)
Path('guard-pid').write_text(str(process.pid))
while True:
    time.sleep(1)
""")
    parent = subprocess.Popen([sys.executable, str(daemon)], cwd=tmp_path)
    pids = []
    try:
        pids = json.loads(wait_file(tmp_path / "worker-pids", parent))
        assert int(wait_file(tmp_path / "detached", parent)) == pids[1]
        with (tmp_path / "ownership.lock").open("r+") as lock:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            parent.kill()
            parent.wait(timeout=5)
            deadline = monotonic() + 5
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    assert monotonic() < deadline
                    sleep(0.02)
            assert all(not os.path.exists(f"/proc/{pid}") for pid in pids)
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
        for pid in pids:
            with suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
