"""Actual authenticated HTTP requests reuse the durable queue and binary store."""

import gc
import hashlib
import http.client
import io
import socket
import ssl
import subprocess
import sys
import weakref
from contextlib import contextmanager, suppress
from dataclasses import replace
from threading import Event, Thread
from time import monotonic
from types import SimpleNamespace

import pytest

from engine_tests.server.test_request_records import IDENTITY, request, store
from engine_tests.training.test_controller import configured
from engine_tests.training.test_prepared_requests import upload
from shadowspill.server import http as http_module
from shadowspill.server.artifacts import ArtifactStore
from shadowspill.server.authentication import BearerTokens
from shadowspill.server.http import HTTP_SCRATCH_BYTES, HTTPService, RunAccess
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.codec import MAX_METADATA_BYTES, decode
from shadowspill_client.protocol.requests import ArtifactRef, OperationRequest
from shadowspill_client.protocol.service import (
    CONTROL_MEDIA_TYPE,
    ServiceError,
    ServiceInfo,
)
from shadowspill_client.transport.http import HTTPTransport, RemoteError, TransportError

TOKEN = "owner-test-" + "a" * 32
OTHER_TOKEN = "other-test-" + "b" * 32
ROOT = "/v1/runs/session/run"


@contextmanager
def endpoint(
    tmp_path,
    *,
    run=None,
    concurrency=3,
    timeout=5,
    tls=None,
    client_tls=None,
    wrong_scope=False,
    host="127.0.0.1",
):
    entered = []
    memory = BufferBudget(HTTP_SCRATCH_BYTES * concurrency, concurrency)
    artifact_memory = BufferBudget(8 << 20, 3)
    with ArtifactStore(
        tmp_path / "artifacts",
        owner_id="owner",
        cache_root=tmp_path,
        memory=artifact_memory,
        max_bytes=4 << 20,
        max_artifacts=10,
    ) as artifacts:

        @contextmanager
        def runs(identity):
            entered.append(identity)
            if run is None or (not wrong_scope and identity != IDENTITY):
                raise KeyError("run")
            yield RunAccess(run.journal, run.controller)

        @contextmanager
        def owner_artifacts(owner):
            entered.append(owner)
            if owner != "owner" and not wrong_scope:
                raise KeyError("owner")
            yield artifacts

        service = HTTPService(
            (host, 0),
            incarnation="server",
            authentication=BearerTokens({TOKEN: "owner", OTHER_TOKEN: "other"}),
            runs=runs,
            artifacts=owner_artifacts,
            memory=memory,
            request_timeout=timeout,
            tls=tls,
        )
        thread = Thread(target=service.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        scheme, hostname = (
            "https" if tls else "http",
            "[::1]" if host == "::1" else host,
        )
        base = f"{scheme}://{hostname}:{service.server_port}"
        budgets = []

        def client(token=TOKEN):
            budget = BufferBudget(HTTP_SCRATCH_BYTES * concurrency, concurrency)
            budgets.append(budget)
            return HTTPTransport(
                base, token=token, memory=budget, timeout=5, ssl_context=client_tls
            )

        try:
            yield SimpleNamespace(
                service=service,
                client=client(),
                connect=client,
                entered=entered,
                memory=memory,
                artifacts=artifacts,
            )
        finally:
            service.shutdown()
            service.server_close()
            thread.join(timeout=5)
            assert not thread.is_alive() and memory.used == (0, 0)
            assert all(budget.used == (0, 0) for budget in budgets)
    assert artifact_memory.used == (0, 0)


def exchange(ctx, method, path, *, headers=(), body=b"", length=None):
    connection = http.client.HTTPConnection(
        "127.0.0.1", ctx.service.server_port, timeout=5
    )
    try:
        connection.putrequest(method, path, skip_accept_encoding=True)
        connection.putheader(
            "Content-Length", str(len(body) if length is None else length)
        )
        for name, value in headers:
            connection.putheader(name, value)
        connection.endheaders(body)
        with connection.getresponse() as response:
            return response.status, decode(response.read())
    finally:
        connection.close()


AUTH = ("Authorization", "Bearer " + TOKEN)


def test_authenticated_owner_is_derived_from_credentials_before_any_lookup(tmp_path):
    with endpoint(tmp_path) as ctx:
        assert ctx.client.request("GET", "/v1/info").value == ServiceInfo(
            "server", "owner"
        )
        assert (
            ctx.connect(OTHER_TOKEN).request("GET", "/v1/info").value.owner_id
            == "other"
        )
        for headers in (
            (),
            (("Authorization", "Basic " + TOKEN),),
            (("Authorization", "Bearer " + "x" * 32),),
            (AUTH, AUTH),
        ):
            status, error = exchange(ctx, "GET", ROOT + "/requests/a", headers=headers)
            assert status == 403 and error.code == "not_authorized"
            assert TOKEN not in str(error) and OTHER_TOKEN not in str(error)
        assert not ctx.entered


def test_real_queue_acceptance_disconnect_deduplication_and_ordered_close(tmp_path):
    with configured(tmp_path, blocked=True) as run, endpoint(tmp_path, run=run) as ctx:
        command, wire = upload(grad_scale=0.5, loss_params={"temperature": 2})
        registered = ctx.client.request("POST", ROOT + "/requests", command).value
        assert registered.state == "uploading" and not run.workers.calls
        accepted = ctx.client.upload(
            ROOT + "/requests/a/input", command.payload, [wire]
        ).value
        assert accepted.acceptance is not None and run.workers.entered.wait(5)
        # Every exchange disconnects. A subsequent handle sees the same accepted work.
        reconnected = ctx.connect()
        pending = reconnected.request(
            "GET", ROOT + "/requests/a?wait=completed&timeout=0.01"
        ).value
        assert pending.state == "running" and pending.acceptance == accepted.acceptance
        assert reconnected.request("POST", ROOT + "/requests", command).value == pending
        with pytest.raises(RemoteError) as conflict:
            reconnected.request(
                "POST", ROOT + "/requests", replace(command, grad_scale=1.0)
            )
        assert conflict.value.status == 400 and run.workers.calls == ["a"]
        run.workers.release.set()
        completed = reconnected.request(
            "GET", ROOT + "/requests/a?wait=completed&timeout=5"
        ).value
        assert completed.result.completed_updates == 1
        assert completed.result.grad_scale == 0.5
        close = OperationRequest(request("close", 2).identity, "close")
        reconnected.request("POST", ROOT + "/requests", close)
        reconnected.request("PUT", ROOT + "/requests/close/input")
        closed = reconnected.request(
            "GET", ROOT + "/requests/close?wait=completed&timeout=5"
        ).value
        run.controller.drain(timeout=5)
        assert (
            closed.state == "completed" and run.controller.closed and run.workers.closed
        )
        assert (
            reconnected.request("POST", ROOT + "/requests", command).value == completed
        )
        assert run.workers.calls == ["a", "close"]


def test_stopped_run_records_can_be_reopened_and_queried_without_reexecution(tmp_path):
    command, wire = upload()
    with configured(tmp_path) as run, endpoint(tmp_path, run=run) as ctx:
        ctx.client.request("POST", ROOT + "/requests", command)
        ctx.client.upload(ROOT + "/requests/a/input", command.payload, [wire])
        completed = ctx.client.request(
            "GET", ROOT + "/requests/a?wait=completed&timeout=5"
        ).value
    with store(tmp_path) as journal:
        run = SimpleNamespace(journal=journal, controller=None)
        with endpoint(tmp_path, run=run) as ctx:
            assert ctx.client.request("GET", ROOT + "/requests/a").value == completed
            assert (
                ctx.client.request("POST", ROOT + "/requests", command).value
                == completed
            )
            with pytest.raises(RemoteError) as unavailable:
                ctx.client.request(
                    "POST", ROOT + "/requests", upload(name="new", sequence=2)[0]
                )
            assert unavailable.value.status == 409 and journal.get("new") is None


def test_writer_replacement_waits_for_prior_work_and_fences_incomplete_uploads(
    tmp_path,
):
    with configured(tmp_path, blocked=True) as run, endpoint(tmp_path, run=run) as ctx:
        a, wire = upload()
        b, _ = upload(name="b", sequence=2)
        ctx.client.request("POST", ROOT + "/requests", a)
        ctx.client.request("POST", ROOT + "/requests", b)
        ctx.client.upload(ROOT + "/requests/a/input", a.payload, [wire])
        assert run.workers.entered.wait(5)
        attached = ctx.client.request("POST", ROOT + "/writers/replacement")
        assert attached.status == 202 and attached.value.writer_epoch == 2
        run.workers.release.set()
        run.controller.drain(timeout=5)
        ready = ctx.client.request("POST", ROOT + "/writers/replacement")
        assert ready.status == 200 and ready.value == attached.value
        assert (
            ctx.client.request("GET", ROOT + "/requests/b").value.failure.code
            == "writer_fenced"
        )
        with pytest.raises(RemoteError) as retired:
            ctx.client.request("POST", ROOT + "/writers/writer")
        assert retired.value.status == 403


def test_owner_and_session_checks_cover_records_and_immutable_binary_downloads(
    tmp_path,
):
    payload = bytes(range(256)) * 1025
    reference = ArtifactRef("output", hashlib.sha256(payload).hexdigest(), len(payload))
    with configured(tmp_path) as run, endpoint(tmp_path, run=run) as ctx:
        ctx.artifacts.put(reference, lambda _: io.BytesIO(payload))
        output = io.BytesIO()
        ctx.client.download("/v1/artifacts/output", reference, output)
        assert output.getvalue() == payload
        for path in (ROOT + "/requests/a", "/v1/artifacts/output"):
            with pytest.raises(RemoteError) as error:
                ctx.connect(OTHER_TOKEN).request("GET", path)
            assert error.value.status == 404
        with pytest.raises(RemoteError) as error:
            ctx.client.request("GET", "/v1/runs/another/run/requests/a")
        assert error.value.status == 404


def test_transport_checks_owner_scope_even_if_a_lookup_returns_the_wrong_owner(
    tmp_path,
):
    with (
        configured(tmp_path) as run,
        endpoint(tmp_path, run=run, wrong_scope=True) as ctx,
    ):
        for path in (ROOT + "/requests/a", "/v1/artifacts/output"):
            with pytest.raises(RemoteError) as error:
                ctx.connect(OTHER_TOKEN).request("GET", path)
            assert error.value.status == 403


@pytest.mark.parametrize(
    "headers,length",
    [
        ((("Content-Length", "0")), 0),
        (("Transfer-Encoding", "chunked"), 0),
        (("Content-Encoding", "gzip"), 0),
        (("Content-Type", CONTROL_MEDIA_TYPE), -1),
        (("Content-Type", CONTROL_MEDIA_TYPE), MAX_METADATA_BYTES + 1),
    ],
)
def test_invalid_metadata_or_framing_is_rejected_without_reading_a_body(
    tmp_path, headers, length
):
    with configured(tmp_path) as run, endpoint(tmp_path, run=run) as ctx:
        status, error = exchange(
            ctx, "POST", ROOT + "/requests", headers=(AUTH, headers), length=length
        )
        assert status == 400 and isinstance(error, ServiceError)
        assert not run.journal.pending() and not run.workers.calls


def test_complete_header_section_is_bounded_before_the_standard_parser(tmp_path):
    with endpoint(tmp_path) as ctx:
        headers = (AUTH, *((f"X-{i}", "x" * 4096) for i in range(20)))
        status, error = exchange(ctx, "GET", "/v1/info", headers=headers)
        assert status == 431 and error.code == "http_error"
        assert not ctx.entered


def test_capacity_is_reserved_before_starting_a_thread_or_reading_headers(tmp_path):
    with endpoint(tmp_path, concurrency=1) as ctx:
        with socket.create_connection(
            ("127.0.0.1", ctx.service.server_port), timeout=5
        ) as blocked:
            blocked.sendall(b"GET /v1/info HTTP/1.1\r\n")
            deadline = monotonic() + 5
            while not ctx.memory.used[1]:
                assert monotonic() < deadline
                Event().wait(0.005)
            with pytest.raises(RemoteError) as error:
                ctx.client.request("GET", "/v1/info")
            assert error.value.status == 503 and error.value.code == "backpressure"
            assert ctx.memory.used == (HTTP_SCRATCH_BYTES, 1)
        assert not ctx.entered


def test_finished_handlers_release_request_headers_without_waiting_for_gc(
    tmp_path, monkeypatch
):
    handlers = []
    setup = http_module._Handler.setup

    def observed(handler):
        handlers.append(weakref.ref(handler))
        setup(handler)

    monkeypatch.setattr(http_module._Handler, "setup", observed)
    gc.collect()
    gc.disable()
    try:
        with endpoint(tmp_path) as ctx:
            for _ in range(4):
                ctx.client.request("GET", "/v1/info")
        assert handlers and all(handler() is None for handler in handlers)
    finally:
        gc.enable()
        gc.collect()


def test_slow_partial_headers_cannot_extend_the_exchange_deadline(tmp_path):
    with endpoint(tmp_path, timeout=0.2) as ctx:
        with socket.create_connection(
            ("127.0.0.1", ctx.service.server_port), timeout=2
        ) as connection:
            started = monotonic()
            connection.sendall(b"GET /v1/info HTTP/1.1\r\nX-Slow: ")
            done = Event()

            def trickle():
                with suppress(OSError):
                    while not done.wait(0.02):
                        connection.sendall(b"x")

            thread = Thread(target=trickle)
            thread.start()
            try:
                assert connection.recv(1024) == b""
                assert monotonic() - started < 1.5
            finally:
                done.set()
                thread.join(timeout=2)
        assert not ctx.entered


def test_truncated_upload_is_recorded_not_started_and_releases_input_capacity(tmp_path):
    with configured(tmp_path) as run, endpoint(tmp_path, run=run) as ctx:
        command, wire = upload()
        ctx.client.request("POST", ROOT + "/requests", command)
        with socket.create_connection(
            ("127.0.0.1", ctx.service.server_port), timeout=5
        ) as connection:
            headers = (
                f"PUT {ROOT}/requests/a/input HTTP/1.1\r\n"
                f"Authorization: Bearer {TOKEN}\r\n"
                f"Content-Type: {command.payload.media_type}\r\n"
                f"Content-Length: {len(wire)}\r\n\r\n"
            )
            connection.sendall(headers.encode("ascii") + wire[:20])
            connection.shutdown(socket.SHUT_WR)
            assert connection.recv(1024)
        terminal = ctx.client.request(
            "GET", ROOT + "/requests/a?wait=completed&timeout=5"
        ).value
        assert terminal.failure.execution_effect == "not_started"
        assert terminal.failure.state_valid and not run.workers.calls
        run.controller.drain(timeout=5)
        assert run.budget.used == (0, 0)


@pytest.mark.parametrize(
    "path",
    [
        "/v1/runs/../run/requests/a",
        "/v1/runs/session%2fother/run/requests/a",
        "/v1/runs/session/run/requests/a?wait=completed&timeout=11",
        "/v1/runs/session/run/requests/a?wait=completed&wait=accepted",
        "/v1/runs/session/run/requests/a?wait=completed&timeout=nan",
    ],
)
def test_paths_and_queries_cannot_supply_filesystem_paths_or_unbounded_waits(
    tmp_path, path
):
    with configured(tmp_path) as run, endpoint(tmp_path, run=run) as ctx:
        ctx.client.request("POST", ROOT + "/requests", upload()[0])
        with pytest.raises(RemoteError) as error:
            ctx.client.request("GET", path)
        assert error.value.status == 400
        # Complete the original upload before fixture teardown.
        command, wire = upload()
        ctx.client.upload(ROOT + "/requests/a/input", command.payload, [wire])


def test_authentication_configuration_retains_digests_and_rejects_unsafe_tokens():
    configured = BearerTokens({TOKEN: "owner", OTHER_TOKEN: "other"})
    assert configured.owner(["bEaReR " + TOKEN]) == "owner"
    assert all(TOKEN.encode() not in pair for pair in configured._owners)
    for token in ("short", "x" * 32 + "\n", "x" * 513):
        with pytest.raises(ValueError):
            BearerTokens({token: "owner"})
    with pytest.raises(ValueError):
        BearerTokens({TOKEN: "../owner"})


def test_http_controller_imports_without_framework_or_catalog():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
for name in ('torch', 'triton', 'pynvml', 'mlops', 'shadowspill_workloads',
             'shadowspill.runtime'):
    sys.modules[name] = None
from shadowspill.server.http import HTTPService, RunAccess
from shadowspill.server.authentication import BearerTokens
""",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


def test_tls_handshake_certificate_verification_and_authenticated_requests(tmp_path):
    certificate, key = tmp_path / "server.pem", tmp_path / "server.key"
    result = subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "1",
            "-subj",
            "/CN=localhost",
            "-addext",
            "subjectAltName=IP:127.0.0.1",
            "-keyout",
            str(key),
            "-out",
            str(certificate),
        ],
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    server = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server.load_cert_chain(certificate, key)
    trusted = ssl.create_default_context(cafile=str(certificate))
    with endpoint(tmp_path, tls=server, client_tls=trusted) as ctx:
        assert ctx.client.request("GET", "/v1/info").value.owner_id == "owner"
        untrusted = HTTPTransport(
            f"https://127.0.0.1:{ctx.service.server_port}",
            token=TOKEN,
            memory=BufferBudget(HTTP_SCRATCH_BYTES, 1),
            timeout=5,
        )
        with pytest.raises(TransportError, match="SSLCertVerificationError"):
            untrusted.request("GET", "/v1/info")


def test_ipv6_loopback_listener_uses_the_same_service(tmp_path):
    if not socket.has_ipv6:
        pytest.skip("host has no IPv6 support")
    with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("::1", 0))
        except OSError:
            pytest.skip("host has no IPv6 loopback route")
    with endpoint(tmp_path, host="::1") as ctx:
        assert ctx.client.request("GET", "/v1/info").value == ServiceInfo(
            "server", "owner"
        )
