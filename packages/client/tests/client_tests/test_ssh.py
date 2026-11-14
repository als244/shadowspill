"""Real local OpenSSH forwarding with private test keys and a borrowed master."""

import hashlib
import io
import os
import pwd
import shutil
import socket
import subprocess
import sys
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from time import monotonic
from types import SimpleNamespace

import pytest

from shadowspill_client import SSH
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.codec import encode
from shadowspill_client.protocol.requests import ArtifactRef
from shadowspill_client.protocol.service import CONTROL_MEDIA_TYPE, ServiceInfo
from shadowspill_client.transport._ssh_process import SSHCommand
from shadowspill_client.transport.http import HTTPTransport, TransportError
from shadowspill_client.transport.ssh import SSH_BYTES, SSHTunnel

TOKEN = "test-token-" + "a" * 32


def available_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def ssh_server():
    daemon = shutil.which("sshd")
    if daemon is None or shutil.which("ssh-keygen") is None:
        pytest.skip("real OpenSSH client, keygen and test daemon are required")
    with TemporaryDirectory(prefix="ss-test-") as directory:
        root = Path(directory)
        for name in ("host", "client"):
            subprocess.run(
                ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(root / name)],
                check=True,
                timeout=5,
            )
        port = available_port()
        (root / "sshd.conf").write_text(
            "\n".join(
                [
                    f"Port {port}",
                    "ListenAddress 127.0.0.1",
                    f"HostKey {root}/host",
                    f"AuthorizedKeysFile {root}/client.pub",
                    f"PidFile {root}/pid",
                    "UsePAM no",
                    "PasswordAuthentication no",
                    "KbdInteractiveAuthentication no",
                    "StrictModes no",
                    "LogLevel ERROR",
                    "AllowTcpForwarding yes",
                ]
            )
            + "\n"
        )
        (root / "known_hosts").write_text(
            f"[127.0.0.1]:{port} " + (root / "host.pub").read_text()
        )
        (root / "client.conf").write_text(
            "\n".join(
                [
                    "Host fixture",
                    "  HostName 127.0.0.1",
                    f"  Port {port}",
                    f"  User {pwd.getpwuid(os.getuid()).pw_name}",
                    f"  IdentityFile {root}/client",
                    f"  UserKnownHostsFile {root}/known_hosts",
                    "  StrictHostKeyChecking yes",
                    "  IdentitiesOnly yes",
                    "  LogLevel ERROR",
                    f"  ControlPath {root}/master",
                    "  ControlMaster auto",
                    "  ControlPersist yes",
                ]
            )
            + "\n"
        )
        command = SSHCommand([daemon, "-D", "-e", "-f", str(root / "sshd.conf")])
        command.start()
        try:
            deadline = monotonic() + 5
            while True:
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.05):
                        break
                except OSError:
                    assert command.returncode is None, command.error.value().decode()
                    assert monotonic() < deadline
                    Event().wait(0.01)
            yield SimpleNamespace(
                root=root, config=str(root / "client.conf"), port=port
            )
        finally:
            command.close()
            command.clear()


@contextmanager
def endpoint():
    data = b"binary service result" * 10000
    reference = ArtifactRef("output", hashlib.sha256(data).hexdigest(), len(data))
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.headers.get("Authorization"))
            assert self.headers.get("Authorization") == "Bearer " + TOKEN
            binary = self.path == "/v1/artifacts/output"
            body = data if binary else encode(ServiceInfo("server", "owner"))
            self.send_response(200)
            self.send_header(
                "Content-Type", reference.media_type if binary else CONTROL_MEDIA_TYPE
            )
            self.send_header("Content-Length", str(len(body)))
            if binary:
                self.send_header("ETag", '"' + reference.sha256 + '"')
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield SimpleNamespace(
            port=server.server_port, data=data, reference=reference, seen=seen
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
        assert not thread.is_alive()


def client(base_url):
    return HTTPTransport(
        base_url, token=TOKEN, memory=BufferBudget(32 << 20, 2), timeout=3
    )


def test_private_foreground_tunnel_carries_authenticated_http_and_binary_data(
    ssh_server,
):
    memory = BufferBudget(SSH_BYTES, 1)
    with (
        endpoint() as service,
        SSHTunnel(
            SSH("fixture", service.port, config_file=ssh_server.config),
            memory=memory,
            timeout=5,
        ) as tunnel,
    ):
        base = tunnel.start()
        master = tunnel._master
        assert master is not None and master.returncode is None
        assert memory.used == (SSH_BYTES, 1)
        transport = client(base)
        assert transport.request("GET", "/v1/info").value == ServiceInfo(
            "server", "owner"
        )
        target = io.BytesIO()
        transport.download("/v1/artifacts/output", service.reference, target)
        assert (
            target.getvalue() == service.data
            and service.seen == ["Bearer " + TOKEN] * 2
        )
        assert not (ssh_server.root / "master").exists()
        process, control, port = master.process, tunnel._control, tunnel._port
    assert tunnel.closed and memory.used == (0, 0) and master.closed
    assert process.returncode is not None and not control.exists()
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout=0.1)
    tunnel.close()


def test_reusing_a_master_cancels_only_the_owned_forward(ssh_server):
    memory = BufferBudget(2 * SSH_BYTES, 2)
    with endpoint() as service:
        master = SSHCommand(
            [
                "ssh",
                "-F",
                ssh_server.config,
                "-N",
                "-T",
                "-M",
                "-o",
                "BatchMode=yes",
                "-o",
                "ControlPersist=no",
                "-o",
                "ForkAfterAuthentication=no",
                "fixture",
            ]
        )
        master.start()
        try:
            deadline = monotonic() + 5
            while not (ssh_server.root / "master").exists():
                assert master.returncode is None and monotonic() < deadline
                Event().wait(0.01)
            settings = SSH("fixture", service.port, config_file=ssh_server.config)
            with (
                SSHTunnel(settings, memory=memory, timeout=5) as first,
                SSHTunnel(settings, memory=memory, timeout=5) as second,
            ):
                first_url, second_url = first.start(), second.start()
                assert (
                    first_url != second_url and first._master is second._master is None
                )
                assert (
                    client(first_url).request("GET", "/v1/info").value.owner_id
                    == "owner"
                )
                first.close()
                assert master.returncode is None and memory.used == (SSH_BYTES, 1)
                assert (
                    client(second_url).request("GET", "/v1/info").value.owner_id
                    == "owner"
                )
                check = subprocess.run(
                    ["ssh", "-F", ssh_server.config, "-O", "check", "fixture"],
                    capture_output=True,
                    timeout=5,
                )
                assert check.returncode == 0, check.stderr
            assert master.returncode is None and memory.used == (0, 0)
        finally:
            master.close()
            master.clear()


def test_host_key_verification_is_preserved_and_failed_start_releases_on_close(
    ssh_server,
):
    (ssh_server.root / "known_hosts").write_text(
        f"[127.0.0.1]:{ssh_server.port} " + (ssh_server.root / "client.pub").read_text()
    )
    memory = BufferBudget(SSH_BYTES, 1)
    with SSHTunnel(
        SSH("fixture", 8000, config_file=ssh_server.config), memory=memory, timeout=5
    ) as tunnel:
        with pytest.raises(RuntimeError, match="HOST IDENTIFICATION HAS CHANGED"):
            tunnel.start()
        assert memory.used == (SSH_BYTES, 1)
    assert memory.used == (0, 0) and tunnel.closed


def test_ssh_start_does_not_claim_the_http_service_is_ready(ssh_server):
    memory = BufferBudget(SSH_BYTES, 1)
    with SSHTunnel(
        SSH("fixture", available_port(), config_file=ssh_server.config),
        memory=memory,
        timeout=5,
    ) as tunnel:
        base = tunnel.start()
        with pytest.raises(TransportError):
            client(base).request("GET", "/v1/info")
        assert tunnel._master.returncode is None and memory.used == (SSH_BYTES, 1)
    assert memory.used == (0, 0)


def test_metadata_budget_is_reserved_before_any_local_process(monkeypatch):
    memory = BufferBudget(SSH_BYTES, 1)
    with (
        memory.reserve(SSH_BYTES),
        SSHTunnel(SSH("host", 8000), memory=memory) as tunnel,
    ):
        monkeypatch.setattr(
            SSHCommand, "start", lambda _: pytest.fail("unreserved SSH command")
        )
        with pytest.raises(TimeoutError):
            tunnel.start()
    assert memory.used == (0, 0)


@pytest.mark.parametrize(
    "changes",
    [
        {"host": "-oProxyCommand=bad"},
        {"host": "user@host; command"},
        {"host": "bad\nhost"},
        {"jump_host": "bad gateway"},
        {"server_host": "host/path"},
        {"server_host": "bad host"},
        {"server_port": 0},
        {"server_port": True},
        {"ssh_port": 65536},
        {"config_file": "bad\0path"},
    ],
)
def test_connection_fields_cannot_become_shell_commands_or_other_options(changes):
    with pytest.raises((TypeError, ValueError)):
        SSH(**{"host": "alias", "server_port": 8000, **changes})


def test_ipv6_target_and_jump_routing_remain_separate_connection_settings():
    settings = SSH(
        "user@host", 8000, server_host="::1", jump_host="user@gateway:2222", ssh_port=22
    )
    assert settings.server_host == "::1" and settings.ssh_port != settings.server_port


def test_local_diagnostics_are_bounded_and_all_owned_processes_are_reaped():
    command = SSHCommand(
        [
            sys.executable,
            "-c",
            "import os; os.write(1, b'a'*(2<<20)); os.write(2,b'b'*(2<<20))",
        ]
    )
    command.start()
    try:
        assert command.wait(5) == 0
        command.close()
        assert command.output.seen == command.error.seen == 2 << 20
        assert len(command.output.value()) == 256 << 10
        assert len(command.error.value()) == 64 << 10
        assert command.process.returncode == 0
    finally:
        command.close()
        command.clear()
    assert command.output.value() == command.error.value() == b""


def test_command_timeout_does_not_reap_before_owned_group_cleanup():
    command = SSHCommand([sys.executable, "-c", "import time; time.sleep(60)"])
    command.start()
    try:
        with pytest.raises(TimeoutError):
            command.wait(0.01)
        assert not command.closed
    finally:
        command.close()
        command.clear()
    assert command.process.returncode is not None and command.closed


def test_failed_local_directory_cleanup_retains_the_connection_budget_for_retry(
    ssh_server, monkeypatch
):
    memory = BufferBudget(SSH_BYTES, 1)
    tunnel = SSHTunnel(
        SSH("fixture", 8000, config_file=ssh_server.config), memory=memory, timeout=5
    )
    tunnel.start()
    cleanup = tunnel._directory.cleanup

    def fail():
        raise OSError("private socket-directory cleanup failed")

    monkeypatch.setattr(tunnel._directory, "cleanup", fail)
    with pytest.raises(OSError):
        tunnel.close()
    assert tunnel._master.closed and not tunnel.closed
    assert memory.used == (SSH_BYTES, 1)
    with pytest.raises(RuntimeError):
        _ = tunnel.base_url
    monkeypatch.setattr(tunnel._directory, "cleanup", cleanup)
    tunnel.close()
    assert tunnel.closed and memory.used == (0, 0)


def test_partial_diagnostic_thread_start_still_reaps_its_owned_child(monkeypatch):
    start = Thread.start
    count = 0

    def fail_second(thread):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("thread creation failed")
        return start(thread)

    monkeypatch.setattr(Thread, "start", fail_second)
    command = SSHCommand([sys.executable, "-c", "import time; time.sleep(60)"])
    with pytest.raises(RuntimeError, match="thread creation"):
        command.start()
    command.close()
    command.clear()
    assert command.closed and command.process.returncode is not None
    assert command.output.source.closed and command.error.source.closed


def test_failed_transient_cleanup_cannot_accumulate_new_commands(monkeypatch):
    memory = BufferBudget(SSH_BYTES, 1)
    tunnel = SSHTunnel(SSH("unused", 8000), memory=memory)
    tunnel._lease = memory.reserve(SSH_BYTES)
    command = tunnel._spawn([sys.executable, "-c", "pass"])
    assert command.wait(5) == 0
    close = command.close

    def fail():
        raise OSError("local command cleanup failed")

    monkeypatch.setattr(command, "close", fail)
    for _ in range(3):
        with pytest.raises(OSError):
            tunnel.close()
        with pytest.raises(RuntimeError, match="retained"):
            tunnel._spawn([sys.executable, "-c", "pass"])
        assert len(tunnel._commands) == 1 and memory.used == (SSH_BYTES, 1)
    monkeypatch.setattr(command, "close", close)
    tunnel.close()
    assert memory.used == (0, 0) and command.closed


def test_borrowed_forward_cleanup_failure_retains_only_its_owner_for_retry(
    ssh_server, monkeypatch
):
    memory = BufferBudget(2 * SSH_BYTES, 2)
    with (
        endpoint() as service,
        SSHTunnel(
            SSH("fixture", service.port, config_file=ssh_server.config),
            memory=memory,
            timeout=5,
        ) as owner,
    ):
        original_url = owner.start()
        path = Path(ssh_server.config)
        path.write_text(
            path.read_text().replace(
                str(ssh_server.root / "master"), str(owner._control)
            )
        )
        borrowed = SSHTunnel(owner.settings, memory=memory, timeout=5)
        forwarded_url = borrowed.start()
        assert borrowed._master is None
        run = borrowed._run

        def fail_cancel(arguments, timeout):
            if "cancel" in arguments:
                return 255, b"", "forward cancellation was not acknowledged"
            return run(arguments, timeout)

        monkeypatch.setattr(borrowed, "_run", fail_cancel)
        try:
            with pytest.raises(RuntimeError, match="forward cleanup"):
                borrowed.close()
            assert not borrowed.closed and memory.used == (2 * SSH_BYTES, 2)
            assert owner._master.returncode is None
            assert (
                client(forwarded_url).request("GET", "/v1/info").value.owner_id
                == "owner"
            )
        finally:
            monkeypatch.setattr(borrowed, "_run", run)
            borrowed.close()
        assert memory.used == (SSH_BYTES, 1)
        assert client(original_url).request("GET", "/v1/info").value.owner_id == "owner"
    assert memory.used == (0, 0)


def test_exited_launcher_cannot_leave_a_proxy_child_holding_diagnostic_pipes(tmp_path):
    if not Path("/proc/self/stat").exists():
        pytest.skip("Linux process-state inspection for orphan proxy qualification")
    marker = tmp_path / "proxy-pid"
    child_code = (
        "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        "time.sleep(60)"
    )
    parent_code = (
        "import subprocess,sys; from pathlib import Path; "
        f"child=subprocess.Popen([sys.executable,'-c',{child_code!r}]); "
        f"Path({str(marker)!r}).write_text(str(child.pid))"
    )
    command = SSHCommand([sys.executable, "-c", parent_code])
    command.start()
    try:
        assert command.wait(5) == 0
        assert not command.closed
    finally:
        command.close()
        command.clear()
    pid = int(marker.read_text())
    path = Path(f"/proc/{pid}/stat")
    deadline = monotonic() + 5
    while True:
        try:
            state = path.read_text().split(")", 1)[1].split()[0]
        except (FileNotFoundError, ProcessLookupError):
            break
        if state == "Z":
            break
        assert monotonic() < deadline
        Event().wait(0.01)
    assert command.closed and command.process.returncode == 0
