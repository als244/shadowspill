"""Linux child supervisor: parent death closes a lifeline, then kills owned work.

This small fresh process retains the node daemon's ownership locks until all descendants
exit, including children that create a new session. It imports no engine runtime.
The command and inherited descriptors come only from installed local launch code.
"""

import ctypes
import os
import select
import signal
import subprocess
import sys
from contextlib import suppress
from time import sleep


def _children() -> tuple[int, ...]:
    with open(f"/proc/self/task/{os.getpid()}/children") as stream:
        return tuple(map(int, stream.read().split()))


def _kill_and_reap() -> None:
    # As a subreaper we adopt descendants when their direct parents die. Never
    # exit (and release inherited ownership locks) while any child remains.
    while children := _children():
        for pid in children:
            try:
                status = os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            except ChildProcessError:
                continue
            if status is None:
                with suppress(ProcessLookupError):
                    if os.getpgid(pid) == pid:
                        os.killpg(pid, signal.SIGKILL)
                    else:
                        os.kill(pid, signal.SIGKILL)
            else:
                # The unreaped child pins its PID until this point. No other
                # thread in this supervisor can reap or recycle it underneath us.
                os.waitpid(pid, 0)
        sleep(0.01)


def run(lifeline: int, inherited: tuple[int, ...], command: list[str]) -> int:
    if not sys.platform.startswith("linux"):
        raise RuntimeError("node daemon process supervision requires Linux subreapers")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0):  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "cannot establish child supervision")
    stopping = False

    def stop(signum: int, frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    process = subprocess.Popen(command, pass_fds=inherited, start_new_session=True)
    # The actual worker alone keeps its private control descriptors. The guard
    # retains only its lifeline and the node daemon ownership-lock descriptors, which
    # are deliberately absent from `inherited`.
    for fd in inherited:
        os.close(fd)
    result = 1
    try:
        while not stopping:
            status = os.waitid(
                os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT
            )
            if status is not None:
                result = (
                    status.si_status
                    if status.si_code == os.CLD_EXITED
                    else 128 + status.si_status
                )
                break
            readable, _, _ = select.select([lifeline], [], [], 0.05)
            if readable:
                # The parent never sends data: either readable condition fences
                # ownership and avoids an indefinitely open, malformed lifeline.
                os.read(lifeline, 1)
                break
    finally:
        _kill_and_reap()
        # We reaped through waitid/waitpid, so Popen must not perform another wait.
        process.returncode = result
        os.close(lifeline)
    return result


if __name__ == "__main__":
    if len(sys.argv) < 5 or sys.argv[3] != "--":
        raise SystemExit("invalid installed supervisor invocation")
    raise SystemExit(
        run(
            int(sys.argv[1]),
            tuple(int(fd) for fd in sys.argv[2].split(",") if fd),
            sys.argv[4:],
        )
    )
