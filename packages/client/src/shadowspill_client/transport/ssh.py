"""An owned loopback forward to the same authenticated HTTP service."""

from __future__ import annotations

import math
import re
import socket
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, RLock
from time import monotonic
from typing import Self

from .._records import integer, number
from ..buffers import BufferBudget, BufferLease
from ._ssh_process import SSHCommand

SSH_BYTES = 4 << 20
_DESTINATION = re.compile(r"[A-Za-z0-9_.@:%+\[\],-]{1,1024}")
_SERVER = re.compile(r"[A-Za-z0-9_.:%-]{1,253}")


@dataclass(frozen=True)
class SSH:
    """SSH settings for reaching a service, passed as ``ssh`` to `ServiceClient`.

    The client forwards a local port to the service with OpenSSH, which runs
    without prompting, so authentication must already work non-interactively.
    Aliases, keys, agent and jump settings come from the OpenSSH
    configuration, and a configured control master for ``host`` is reused
    while it runs. These settings stay local and are not saved with any run.

    Parameters
    ----------
    host : str
        OpenSSH destination or alias of the machine that reaches the service.
    server_port : int
        The service's port, as reached from ``host``.
    server_host : str, default "127.0.0.1"
        The service's host name or IP address, as reached from ``host``.
    jump_host : str, optional
        OpenSSH destination to jump through on the way to ``host``.
    ssh_port : int, optional
        SSH port of ``host``. By default, the OpenSSH configuration's.
    config_file : str, optional
        OpenSSH configuration file. By default, the user's.

    Raises
    ------
    ValueError
        If a value is not a valid destination, host, port or path.
    """

    host: str
    server_port: int
    server_host: str = "127.0.0.1"
    jump_host: str | None = None
    ssh_port: int | None = None
    config_file: str | None = None

    def __post_init__(self) -> None:
        for name in ("host", "jump_host"):
            value = getattr(self, name)
            if value is None and name == "jump_host":
                continue
            if (
                not isinstance(value, str)
                or not _DESTINATION.fullmatch(value)
                or value.startswith("-")
            ):
                raise ValueError(f"{name} requires an OpenSSH destination or alias")
        if not isinstance(self.server_host, str) or not _SERVER.fullmatch(
            self.server_host
        ):
            raise ValueError("server_host requires a hostname or IP address")
        for name in ("server_port", "ssh_port"):
            value = getattr(self, name)
            if value is not None:
                if integer(value, name, minimum=1) > 65535:
                    raise ValueError(f"{name} exceeds 65535")
            elif name == "server_port":
                raise ValueError("server_port is required")
        if self.config_file is not None and (
            not isinstance(self.config_file, str)
            or not self.config_file
            or "\0" in self.config_file
        ):
            raise ValueError("config_file requires a local OpenSSH configuration path")


class SSHTunnel:
    """Own one forward, never an allocation or a training run.

    Reuse an existing configured master with forward/cancel control commands.
    Otherwise start one private foreground master and own its process group.
    Startup confirms only SSH forwarding; the caller separately authenticates
    and checks HTTP service readiness. There is no reconnect or mutation retry.
    Keep this owner if start/close fails so cleanup can be retried explicitly.
    """

    def __init__(self, settings: SSH, *, memory: BufferBudget, timeout: float = 30):
        if not isinstance(settings, SSH):
            raise TypeError("SSH tunnel requires SSH connection settings")
        self.timeout = number(timeout, "SSH connection timeout", minimum=0)
        if not self.timeout or self.timeout > 300:
            raise ValueError("SSH connection timeout must be in (0, 300]")
        if memory.max_bytes < SSH_BYTES:
            raise ValueError("SSH connection does not fit the client memory budget")
        self.settings, self._memory = settings, memory
        self._lease: BufferLease | None = None
        self._directory: TemporaryDirectory[str] | None = None
        self._commands: list[SSHCommand] = []
        self._master: SSHCommand | None = None
        self._control: Path | None = None
        self._socket_identity: tuple[int, int] | None = None
        self._forward: str | None = None
        self._port: int | None = None
        self._started = self._closing = self.closed = False
        self._lock = RLock()

    @property
    def base_url(self) -> str:
        if self._closing or self._port is None:
            raise RuntimeError("SSH forward is not available")
        return f"http://127.0.0.1:{self._port}"

    def _options(self) -> list[str]:
        result = [
            "ssh",
            "-T",
            "-N",
            "-a",
            "-x",
            "-o",
            "BatchMode=yes",
            "-o",
            "ForkAfterAuthentication=no",
            "-o",
            "ControlPersist=no",
            "-o",
            "ClearAllForwardings=yes",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            f"ConnectTimeout={math.ceil(self.timeout)}",
        ]
        if self.settings.config_file is not None:
            result.extend(("-F", self.settings.config_file))
        if self.settings.ssh_port is not None:
            result.extend(("-p", str(self.settings.ssh_port)))
        if self.settings.jump_host is not None:
            result.extend(("-J", self.settings.jump_host))
        return result

    def _spawn(self, arguments: list[str]) -> SSHCommand:
        if any(command is not self._master for command in self._commands):
            raise RuntimeError("retained OpenSSH command cleanup requires retry")
        command = SSHCommand(arguments)
        self._commands.append(command)
        command.start()
        return command

    def _run(self, arguments: list[str], timeout: float) -> tuple[int, bytes, str]:
        command = self._spawn(arguments)
        try:
            result = command.wait(timeout)
            command.close()
            assert command.output is not None and command.error is not None
            if (
                command.output.seen > command.output.limit
                or command.output.failed
                or command.error.failed
            ):
                raise RuntimeError("OpenSSH diagnostics exceed their bounded envelope")
            return (
                result,
                command.output.value(),
                command.error.value().decode(errors="replace")[:1024],
            )
        finally:
            command.close()
            command.clear()
            self._commands.remove(command)

    def _control_command(self, action: str, *, forward: str | None = None) -> list[str]:
        assert self._control is not None
        # No user-configured forwards or route changes may enter a local control
        # request. The explicit socket pins the already authenticated connection.
        result = ["ssh", "-F", "none", "-S", str(self._control), "-O", action]
        if forward is not None:
            result.extend(("-L", forward))
        result.append(self.settings.host)
        return result

    def _identity(self) -> tuple[int, int] | None:
        assert self._control is not None
        try:
            facts = self._control.stat()
        except FileNotFoundError:
            return None
        return facts.st_dev, facts.st_ino

    def _remaining(self, deadline: float) -> float:
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError("OpenSSH forwarding exceeded its startup deadline")
        return remaining

    def start(self) -> str:
        with self._lock:
            if self._started or self.closed:
                raise RuntimeError("SSH tunnel can only be started once")
            self._started = True
            self._lease = self._memory.reserve(SSH_BYTES, timeout=0)
            deadline = monotonic() + self.timeout
            code, configuration, error = self._run(
                [*self._options(), "-G", self.settings.host], self._remaining(deadline)
            )
            if code:
                raise RuntimeError("OpenSSH configuration failed: " + error)
            control = next(
                (
                    line[12:].strip()
                    for line in configuration.decode().splitlines()
                    if line.startswith("controlpath ")
                ),
                "none",
            )
            del configuration
            if control != "none":
                self._control = Path(control).expanduser()
                self._socket_identity = self._identity()
                code, _, _ = self._run(
                    self._control_command("check"), self._remaining(deadline)
                )
                if code:
                    self._socket_identity = None
            if self._socket_identity is None:
                # A short private path also fits macOS Unix-domain socket limits.
                self._directory = TemporaryDirectory(prefix="ss-ssh-", dir="/tmp")
                self._control = Path(self._directory.name) / "control"
                self._master = self._spawn(
                    [
                        *self._options(),
                        "-M",
                        "-S",
                        str(self._control),
                        self.settings.host,
                    ]
                )
                while True:
                    if self._master.returncode is not None:
                        self._master.close()
                        assert self._master.error is not None
                        error = self._master.error.value().decode(errors="replace")[
                            :1024
                        ]
                        raise RuntimeError(
                            "OpenSSH connection failed; use normal ssh if interactive "
                            "authentication is required: " + error
                        )
                    self._socket_identity = self._identity()
                    if self._socket_identity is not None:
                        code, _, _ = self._run(
                            self._control_command("check"), self._remaining(deadline)
                        )
                        if not code:
                            break
                    Event().wait(min(0.02, self._remaining(deadline)))
            host = self.settings.server_host
            if ":" in host:
                host = f"[{host}]"
            for _ in range(4):
                if self._identity() != self._socket_identity:
                    raise RuntimeError("OpenSSH master changed before forwarding")
                with socket.socket() as listener:
                    listener.bind(("127.0.0.1", 0))
                    port = listener.getsockname()[1]
                self._forward = f"127.0.0.1:{port}:{host}:{self.settings.server_port}"
                code, _, error = self._run(
                    self._control_command("forward", forward=self._forward),
                    self._remaining(deadline),
                )
                if not code:
                    self._port = port
                    return self.base_url
                self._forward = None
                if (
                    "cannot listen to port" not in error
                    and "Address already in use" not in error
                ):
                    raise RuntimeError("OpenSSH forwarding failed: " + error)
            raise RuntimeError("OpenSSH could not claim an available loopback port")

    def close(self) -> None:
        with self._lock:
            if self.closed:
                return
            self._closing = True
            # Retire failed transient commands before issuing another control
            # request. Retries cannot accumulate diagnostic/thread owners.
            for command in tuple(self._commands):
                if command is not self._master:
                    command.close()
                    command.clear()
                    self._commands.remove(command)
            if self._master is None and self._forward is not None:
                # A replacement master at the same pathname does not own this
                # forward. Never cancel another connection's resources.
                if self._identity() == self._socket_identity:
                    code, _, error = self._run(
                        self._control_command("cancel", forward=self._forward),
                        self.timeout,
                    )
                    if code:
                        check, _, _ = self._run(
                            self._control_command("check"), self.timeout
                        )
                        if not check:
                            raise RuntimeError(
                                "OpenSSH forward cleanup failed: " + error
                            )
                self._forward = None
            for command in tuple(self._commands):
                command.close()
                command.clear()
                self._commands.remove(command)
            if self._directory is not None:
                self._directory.cleanup()
            if self._lease is not None:
                self._lease.close()
            self._port = None
            self.closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
