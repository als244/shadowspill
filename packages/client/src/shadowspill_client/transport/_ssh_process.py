"""Bounded local OpenSSH command ownership, without the engine dependency."""

from __future__ import annotations

import os
import signal
import subprocess
from collections.abc import Sequence
from contextlib import suppress
from threading import Event, Lock, RLock, Thread
from typing import IO


class _Capture:
    def __init__(self, source: IO[bytes], limit: int) -> None:
        self.source, self.limit = source, limit
        self._data = bytearray()
        self.seen = 0
        self.failed = False
        self._lock = Lock()
        self.thread = Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        try:
            while block := self.source.read(8192):
                with self._lock:
                    self.seen += len(block)
                    self._data.extend(block[: max(0, self.limit - len(self._data))])
                del block
        except Exception:
            self.failed = True
        finally:
            self.source.close()

    def value(self) -> bytes:
        with self._lock:
            return bytes(self._data)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


class SSHCommand:
    """One foreground process group; a private master never daemonizes.

    Observe the leader without reaping it until group cleanup, including proxy
    children, is confirmed. This local connection owner has bounded in-memory
    diagnostics; it does not import worker launch, node or GPU orchestration.
    """

    def __init__(self, command: Sequence[str]) -> None:
        self.command = tuple(command)
        self.process: subprocess.Popen[bytes] | None = None
        self.output: _Capture | None = None
        self.error: _Capture | None = None
        self.returncode: int | None = None
        self._lost = self._reaped = self.closed = False
        self._exited, self._lock = Event(), RLock()
        self._watcher = Thread(target=self._observe, daemon=True)

    def start(self) -> None:
        if self.process is not None or self.closed:
            raise RuntimeError("SSH command can only start once")
        if os.name != "posix" or not hasattr(os, "WNOWAIT"):
            raise RuntimeError("owned OpenSSH requires POSIX non-reaping observation")
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            start_new_session=True,
        )
        assert self.process.stdout is not None and self.process.stderr is not None
        self.output = _Capture(self.process.stdout, 256 << 10)
        self.error = _Capture(self.process.stderr, 64 << 10)
        try:
            self.output.thread.start()
            self.error.thread.start()
            self._watcher.start()
        except BaseException:
            self._signal(signal.SIGKILL)
            self.process.wait(timeout=5)
            self._reaped = True
            self._exited.set()
            for capture in (self.output, self.error):
                if capture.thread.ident is None:
                    capture.source.close()
            raise

    def _observe(self) -> None:
        assert self.process is not None
        try:
            result = os.waitid(os.P_PID, self.process.pid, os.WEXITED | os.WNOWAIT)
            assert result is not None
            self.returncode = (
                result.si_status
                if result.si_code == os.CLD_EXITED
                else -result.si_status
            )
        except Exception:
            self._lost = True
        finally:
            self._exited.set()

    def wait(self, timeout: float) -> int:
        if not self._exited.wait(timeout):
            raise TimeoutError("OpenSSH command exceeded its connection deadline")
        if self._lost:
            raise RuntimeError("OpenSSH process ownership was lost")
        assert self.returncode is not None
        return self.returncode

    def _signal(self, value: signal.Signals) -> None:
        if self.process is not None and not self._reaped and not self._lost:
            with suppress(ProcessLookupError):
                os.killpg(self.process.pid, value)

    def close(self) -> None:
        with self._lock:
            if self.closed:
                return
            if self.process is not None:
                if self._lost:
                    raise RuntimeError("OpenSSH process ownership was lost")
                if not self._reaped:
                    if not self._exited.is_set():
                        self._signal(signal.SIGTERM)
                        self._exited.wait(2)
                    # Keep the leader's PID unreaped through descendant cleanup,
                    # even when it exited first and left inherited pipes open.
                    self._signal(signal.SIGKILL)
                    self.wait(5)
                    self.process.wait(timeout=5)
                    self._reaped = True
                if self._watcher.ident is not None:
                    self._watcher.join(5)
                for capture in (self.output, self.error):
                    if capture is not None and capture.thread.ident is not None:
                        capture.thread.join(5)
                        if capture.thread.is_alive():
                            raise RuntimeError("OpenSSH diagnostic pipes remain owned")
            self.closed = True

    def clear(self) -> None:
        if not self.closed:
            raise RuntimeError("close OpenSSH before clearing its diagnostic owner")
        for capture in (self.output, self.error):
            if capture is not None:
                capture.clear()
