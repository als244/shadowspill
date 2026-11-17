"""Darwin process ownership where Python does not expose POSIX waitid."""

from __future__ import annotations

import ctypes
import errno
import os
import subprocess


class _SignalValue(ctypes.Union):
    _fields_ = [("integer", ctypes.c_int), ("pointer", ctypes.c_void_p)]


class _SignalInfo(ctypes.Structure):
    # Darwin's public siginfo_t ABI from sys/signal.h, including reserved space.
    _fields_ = [
        ("signo", ctypes.c_int),
        ("error", ctypes.c_int),
        ("code", ctypes.c_int),
        ("pid", ctypes.c_int),
        ("uid", ctypes.c_uint),
        ("status", ctypes.c_int),
        ("address", ctypes.c_void_p),
        ("value", _SignalValue),
        ("band", ctypes.c_long),
        ("reserved", ctypes.c_ulong * 7),
    ]


def wait_for_exit(pid: int) -> int:
    """Observe an owned child's exit with WNOWAIT, preserving its PID."""
    waitid = ctypes.CDLL(None, use_errno=True).waitid
    waitid.argtypes = [
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.POINTER(_SignalInfo),
        ctypes.c_int,
    ]
    waitid.restype = ctypes.c_int
    info = _SignalInfo()
    while waitid(os.P_PID, pid, ctypes.byref(info), os.WEXITED | os.WNOWAIT):
        error = ctypes.get_errno()
        if error != errno.EINTR:
            raise OSError(error, os.strerror(error))
    if info.pid != pid:
        raise RuntimeError("waitid returned an unexpected child")
    return int(info.status if info.code == os.CLD_EXITED else -info.status)


def group_exited(pid: int) -> bool:
    """Distinguish a zombie-only group from a real killpg permission failure."""
    result = subprocess.run(
        ["/bin/ps", "-g", str(pid), "-o", "stat="],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=5,
        check=True,
    )
    states = result.stdout.split()
    return bool(states) and all(state.startswith(b"Z") for state in states)
