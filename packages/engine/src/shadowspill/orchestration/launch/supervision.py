"""Own fresh POSIX process groups and bounded logs without importing a framework."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, RLock, Thread
from typing import IO, BinaryIO, Self

from shadowspill_client._records import integer, number

_CHUNK = 64 << 10


@dataclass(frozen=True)
class LogStatus:
    bytes_seen: int
    bytes_written: int
    limit_bytes: int
    error: str | None

    @property
    def truncated(self) -> bool:
        return self.bytes_seen > self.limit_bytes or self.error is not None


class _LogDrain:
    def __init__(self, source: IO[bytes], destination: BinaryIO, limit: int) -> None:
        self.source, self.destination, self.limit = source, destination, limit
        self._seen = self._written = 0
        self._error: str | None = None
        self._lock = Lock()
        self.thread = Thread(target=self._run, daemon=True)

    @property
    def status(self) -> LogStatus:
        with self._lock:
            return LogStatus(self._seen, self._written, self.limit, self._error)

    def _run(self) -> None:
        try:
            while block := self.source.read(_CHUNK):
                try:
                    with self._lock:
                        self._seen += len(block)
                        remaining = 0 if self._error else self.limit - self._written
                    part = memoryview(block)[:remaining]
                    try:
                        while part:
                            written = self.destination.write(part)
                            if type(written) is not int or not 0 < written <= len(part):
                                raise OSError("worker log write made no progress")
                            with self._lock:
                                self._written += written
                            part = part[written:]
                    except Exception as error:
                        # Keep draining after disk failure/limit so logging cannot
                        # block the child or retain an exception's chunk buffers.
                        with self._lock:
                            self._error = (str(error).strip() or type(error).__name__)[
                                :1024
                            ]
                    finally:
                        del part
                finally:
                    del block
        except Exception as error:
            with self._lock:
                self._error = (str(error).strip() or type(error).__name__)[:1024]
        finally:
            self.source.close()
            self.destination.close()


class OwnedProcess:
    """A fresh child session, explicit environment and one bounded log per stream.

    Commands come from trusted installed launch code (for example worker_command),
    never request-provided executable names. The child inherits no stdin; private
    worker control uses its own stream. Lifetime has no idle timeout. Close first
    allows orderly exit after the owner closes that control stream, then escalates
    only this owned process group. It does not release a scheduler allocation.
    """

    def __init__(
        self,
        command: Sequence[str],
        *,
        environment: Mapping[str, str],
        directory: Path,
        cwd: Path,
        log_limit_bytes: int = 64 << 20,
        pass_fds: Sequence[int] = (),
        guard_fds: Sequence[int] | None = None,
    ) -> None:
        if os.name != "posix" or not hasattr(os, "WNOWAIT"):
            raise RuntimeError(
                "worker ownership requires POSIX non-reaping exit observation"
            )
        if (
            isinstance(command, (str, bytes))
            or not command
            or any(not isinstance(arg, str) or not arg for arg in command)
        ):
            raise ValueError("provide a nonempty installed worker command")
        integer(log_limit_bytes, "per-stream log byte limit")
        if guard_fds is not None and not sys.platform.startswith("linux"):
            raise RuntimeError("agent child supervision requires Linux")
        if guard_fds is not None and (
            not isinstance(guard_fds, (tuple, list))
            or len(guard_fds) > 1025
            or any(type(fd) is not int or fd < 3 for fd in guard_fds)
            or len(set(guard_fds)) != len(guard_fds)
            or set(guard_fds).intersection(pass_fds)
        ):
            raise ValueError("supervisor ownership descriptors must be distinct")
        if (
            not isinstance(pass_fds, (tuple, list))
            or len(pass_fds) > 32
            or any(type(fd) is not int or fd < 3 for fd in pass_fds)
            or len(set(pass_fds)) != len(pass_fds)
        ):
            raise ValueError(
                "pass_fds must name at most 32 distinct private descriptors"
            )
        directory.mkdir(mode=0o700)
        output = (directory / "stdout.log").open("xb", buffering=0)
        try:
            error = (directory / "stderr.log").open("xb", buffering=0)
        except BaseException:
            output.close()
            raise
        try:
            self._lifeline: int | None = None
            self._guarded = guard_fds is not None
            child_lifeline = None
            if guard_fds is not None:
                from .processes import worker_command

                child_lifeline, self._lifeline = os.pipe()
                command = worker_command(
                    "shadowspill.orchestration.launch.parent_guard",
                    (child_lifeline, ",".join(map(str, pass_fds)), "--", *command),
                )
                pass_fds = (*pass_fds, child_lifeline, *guard_fds)
            self._process = subprocess.Popen(
                list(command),
                env=dict(environment),
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                start_new_session=True,
                pass_fds=tuple(pass_fds),
            )
        except BaseException:
            if self._lifeline is not None:
                os.close(self._lifeline)
                self._lifeline = None
            output.close()
            error.close()
            raise
        finally:
            if child_lifeline is not None:
                os.close(child_lifeline)
        self.pid = self._process.pid
        self._command = tuple(command)
        self._reaped = False
        self._returncode: int | None = None
        self._watch_error: str | None = None
        self._close_lock = RLock()
        self._exited = Event()
        self._watcher = Thread(target=self._observe_exit, daemon=True)
        assert self._process.stdout is not None and self._process.stderr is not None
        self._logs = (
            _LogDrain(self._process.stdout, output, log_limit_bytes),
            _LogDrain(self._process.stderr, error, log_limit_bytes),
        )
        self.closed = False
        try:
            for log in self._logs:
                log.thread.start()
            self._watcher.start()
        except BaseException:
            self._signal(signal.SIGKILL)
            self._process.wait(timeout=5)
            self._reaped = True
            for log in self._logs:
                if log.thread.ident is None:
                    log.source.close()
                    log.destination.close()
                else:
                    log.thread.join(timeout=5)
            self.closed = True
            raise

    @property
    def returncode(self) -> int | None:
        if self._watch_error is not None:
            raise RuntimeError(self._watch_error)
        return self._returncode

    def _observe_exit(self) -> None:
        try:
            # Keep the leader waitable until close has killed any descendants.
            # Its unreaped PID cannot be recycled into an unrelated group.
            status = os.waitid(os.P_PID, self.pid, os.WEXITED | os.WNOWAIT)
            assert status is not None
            self._returncode = (
                status.si_status
                if status.si_code == os.CLD_EXITED
                else -status.si_status
            )
        except Exception as error:
            # If somebody else reaped the child, group ownership can no longer
            # be proved. Never signal a potentially recycled group identifier.
            self._reaped = True
            self._watch_error = f"worker exit ownership lost: {type(error).__name__}"
        finally:
            self._exited.set()

    @property
    def logs(self) -> tuple[LogStatus, LogStatus]:
        return self._logs[0].status, self._logs[1].status

    def wait(self, timeout: float | None = None) -> int:
        if timeout is not None:
            number(timeout, "worker wait timeout", minimum=0)
        if not self._exited.wait(timeout):
            assert timeout is not None
            raise subprocess.TimeoutExpired(self._command, timeout)
        result = self.returncode
        assert result is not None
        return result

    def _signal(self, sig: signal.Signals) -> None:
        if self._reaped:
            return
        if self._guarded and sig == signal.SIGKILL:
            # Never kill the cleanup owner while descendants might still live.
            # Closing the lifeline asks it to kill/reap all owned work. A stuck
            # supervisor keeps its locks and resource claim for explicit retry.
            if self._lifeline is not None:
                os.close(self._lifeline)
                self._lifeline = None
            sig = signal.SIGTERM
        with suppress(ProcessLookupError):
            # start_new_session pins the child's group to its owned PID. Never
            # infer another process group from allocation/global rank metadata.
            os.killpg(self.pid, sig)

    def close(self, *, grace_seconds: float = 5, terminate_seconds: float = 2) -> None:
        number(grace_seconds, "orderly worker exit timeout", minimum=0)
        number(terminate_seconds, "worker termination timeout", minimum=0)
        with self._close_lock:
            self._close_owned(grace_seconds, terminate_seconds)

    def _close_owned(self, grace_seconds: float, terminate_seconds: float) -> None:
        if self.closed:
            return
        try:
            try:
                self.wait(timeout=grace_seconds)
            except subprocess.TimeoutExpired:
                self._signal(signal.SIGTERM)
                try:
                    self.wait(timeout=terminate_seconds)
                except subprocess.TimeoutExpired:
                    self._signal(signal.SIGKILL)
                    self.wait(timeout=5)
        finally:
            # A launcher may exit before a child that inherited its log pipes.
            # No descendant of this owned group may outlive its agent owner.
            self._signal(signal.SIGKILL)
            if not self._exited.wait(5):
                raise RuntimeError("owned worker did not report termination")
            self._process.wait(timeout=5)
            self._reaped = True
            self._watcher.join(timeout=5)
            for log in self._logs:
                log.thread.join(timeout=5)
            if any(log.thread.is_alive() for log in self._logs):
                raise RuntimeError("owned worker log streams did not close")
            self.closed = True
            if self._lifeline is not None:
                os.close(self._lifeline)
                self._lifeline = None

    def __enter__(self) -> Self:
        if self.closed:
            raise RuntimeError("worker process is closed")
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
