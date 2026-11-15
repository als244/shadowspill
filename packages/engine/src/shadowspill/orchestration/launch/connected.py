"""Agent-owned child control sockets, separate from bounded process logs."""

from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from socket import SHUT_RDWR, socketpair
from threading import RLock
from typing import BinaryIO, Self, cast

from shadowspill_client._records import number

from .supervision import OwnedProcess


class ConnectedProcess:
    """Pass one private socket descriptor to a trusted installed child command.

    ``command(fd)`` is agent code, not a request-supplied callable/executable. The
    child owns that descriptor; no listener, TCP port or authentication token is
    exposed. Agents use authenticated connections between hosts separately.
    """

    def __init__(
        self,
        command: Callable[[int], Sequence[str]],
        *,
        environment: Mapping[str, str],
        directory: Path,
        cwd: Path,
        log_limit_bytes: int = 64 << 20,
        startup_timeout: float = 120,
        shutdown_grace_seconds: float = 5,
    ) -> None:
        if number(startup_timeout, "worker startup timeout", minimum=0) == 0:
            raise ValueError("worker startup timeout must be positive")
        self._shutdown_grace = number(
            shutdown_grace_seconds, "worker shutdown grace", minimum=0
        )
        self._socket, child = socketpair()
        self.closed = False
        self._lock = RLock()
        self._ready = False
        try:
            self._socket.settimeout(startup_timeout)
            self.stream = cast(BinaryIO, self._socket.makefile("rwb", buffering=0))
            try:
                self.process = OwnedProcess(
                    command(child.fileno()),
                    environment=environment,
                    directory=directory,
                    cwd=cwd,
                    log_limit_bytes=log_limit_bytes,
                    pass_fds=(child.fileno(),),
                )
            except BaseException:
                self.stream.close()
                raise
        except BaseException:
            self._socket.close()
            raise
        finally:
            # Only the child may retain its socket end after spawn; otherwise
            # process death could leave a parent-owned end hiding EOF forever.
            child.close()

    def ready(self) -> None:
        """Remove only the startup deadline after the caller verifies readiness."""
        with self._lock:
            if self.closed or self._ready or self.stream.closed:
                raise RuntimeError("worker channel is closed or already ready")
            self._socket.settimeout(None)
            self._ready = True

    def disconnect(self) -> None:
        """Close private I/O while retaining ownership of the running process."""
        with self._lock:
            # Shutdown wakes blocked peer/local reads before waiting for the child.
            with suppress(OSError):
                self._socket.shutdown(SHUT_RDWR)
            try:
                self.stream.close()
            finally:
                self._socket.close()

    def close(
        self, *, grace_seconds: float | None = None, terminate_seconds: float = 2
    ) -> None:
        """Use the launcher's bounded finalization allowance unless overridden."""
        with self._lock:
            if not self.closed:
                try:
                    self.disconnect()
                finally:
                    self.process.close(
                        grace_seconds=self._shutdown_grace
                        if grace_seconds is None
                        else grace_seconds,
                        terminate_seconds=terminate_seconds,
                    )
                self.closed = True

    def __enter__(self) -> Self:
        if self.closed:
            raise RuntimeError("connected child is closed")
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
