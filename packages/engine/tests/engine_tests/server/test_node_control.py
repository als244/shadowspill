"""Actual HTTP registration, bounded long polling and independent liveness."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from threading import Event, Thread
from time import monotonic, sleep

import pytest

from engine_tests.orchestration.test_node_registry import ACCESS, TOKEN, join_request
from shadowspill.orchestration.nodes.authentication import NodeTokens
from shadowspill.orchestration.nodes.membership import Membership
from shadowspill.orchestration.nodes.protocol import HTTP_BYTES, Heartbeat
from shadowspill.orchestration.nodes.registry import NodeRegistry, NodeUnavailable
from shadowspill.orchestration.nodes.transport import NodeTransport
from shadowspill.server.authentication import BearerTokens
from shadowspill.server.http import HTTP_SCRATCH_BYTES, HTTPService
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.transport.http import HTTPTransport, RemoteError

CLIENT_TOKEN = "client-test-" + "b" * 32


@contextmanager
def endpoint():
    registry = NodeRegistry(
        "server", heartbeat_seconds=0.05, lease_seconds=0.5, poll_seconds=0.4
    )
    memory = BufferBudget(HTTP_SCRATCH_BYTES, 1)
    connections = BufferBudget(HTTP_BYTES * 6, 6)
    server = HTTPService(
        ("127.0.0.1", 0),
        incarnation="server",
        authentication=BearerTokens({CLIENT_TOKEN: "owner"}),
        runs=lambda _: nullcontext(),
        artifacts=lambda _: nullcontext(),
        memory=memory,
        connections=connections,
        nodes=registry,
        node_authentication=NodeTokens({TOKEN: ACCESS}),
        hardware=registry.inventory,
        request_timeout=3,
    )
    thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        yield registry, server, url
    finally:
        registry.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert memory.used == connections.used == (0, 0)


def test_node_credentials_have_no_client_privileges_and_client_cannot_join():
    with endpoint() as (_, _, url):
        with pytest.raises(NodeUnavailable, match="not_authorized"):
            NodeTransport(url, CLIENT_TOKEN).exchange("join", join_request())
        transport = HTTPTransport(
            url, token=TOKEN, memory=BufferBudget(HTTP_SCRATCH_BYTES, 1)
        )
        with pytest.raises(RemoteError):
            transport.request("GET", "/v1/info")
        joined = NodeTransport(url, TOKEN).exchange("join", join_request())
        assert joined.scope.controller == "server"


def test_lost_receipt_ack_is_deduplicated_and_results_do_not_wait_for_poll():
    class InterruptedTransport(NodeTransport):
        lost = False

        def exchange(self, route, request):
            result = super().exchange(route, request)
            if route == "receipt" and not self.lost:
                self.lost = True
                raise OSError("reply lost after acceptance")
            return result

    with (
        endpoint() as (registry, server, url),
        ThreadPoolExecutor(max_workers=1) as pool,
    ):
        membership = Membership(InterruptedTransport(url, TOKEN))
        joined, cleaned = Event(), Event()
        executed = []

        def handle(command):
            executed.append(command.sequence)
            return {"ok": True}

        future = pool.submit(
            membership.run,
            join_request(),
            handle=handle,
            cleanup=cleaned.set,
            assignments=lambda: (),
            joined=lambda _: joined.set(),
        )
        try:
            assert joined.wait(2)
            for index in range(40):
                started = monotonic()
                command = registry.submit(
                    "worker",
                    "run",
                    "claim" if index == 0 else "worker",
                    {"devices": {"gpu": 90}} if index == 0 else {},
                )
                assert registry.wait("worker", command, timeout=2).error is None
                registry.retire("worker", command)
                # Polling is 400 ms: completion has its own outbound lane.
                assert monotonic() - started < 0.35
            assert executed == list(range(1, 41))
            assert server.memory.used == (0, 0)
            client = HTTPTransport(
                url, token=CLIENT_TOKEN, memory=BufferBudget(HTTP_SCRATCH_BYTES, 1)
            )
            assert client.request("GET", "/v1/hardware").value.nodes[0].id == "worker"
        finally:
            membership.stop.set()
        future.result(timeout=3)
        assert cleaned.is_set() and not registry.inventory().nodes


def test_long_command_renews_membership_and_stop_has_independent_lane():
    with endpoint() as (registry, _, url), ThreadPoolExecutor(max_workers=1) as pool:
        membership = Membership(NodeTransport(url, TOKEN), max_commands=2)
        joined, running, released = Event(), Event(), Event()

        def handle(command):
            if command.operation == "claim":
                running.set()
                assert released.wait(3)
            elif command.operation == "stop":
                released.set()
            return {}

        future = pool.submit(
            membership.run,
            join_request(),
            handle=handle,
            cleanup=released.set,
            assignments=lambda: (),
            joined=lambda _: joined.set(),
        )
        try:
            assert joined.wait(2)
            claim = registry.submit("worker", "run", "claim", {"devices": {"gpu": 90}})
            assert running.wait(1)
            sleep(0.8)  # Longer than a full lease, with work still blocked.
            assert registry.inventory().nodes
            stop = registry.submit("worker", "run", "stop")
            assert registry.wait("worker", stop, timeout=1).error is None
            assert registry.wait("worker", claim, timeout=1).error is None
        finally:
            membership.stop.set()
        future.result(timeout=3)


def test_late_heartbeat_reply_cannot_extend_expired_local_authority():
    class Delayed(NodeTransport):
        def exchange(self, route, request):
            result = super().exchange(route, request)
            if isinstance(request, Heartbeat):
                sleep(0.65)
            return result

    with endpoint() as (registry, _, url):
        cleaned = Event()
        membership = Membership(Delayed(url, TOKEN))
        with pytest.raises(NodeUnavailable):
            membership.run(
                join_request(),
                handle=lambda _: {},
                cleanup=cleaned.set,
                assignments=lambda: (),
                joined=lambda _: None,
            )
        assert cleaned.is_set()
        assert not registry.inventory().nodes


def test_remote_control_requires_verified_tls():
    import ssl

    with pytest.raises(ValueError, match="HTTPS"):
        NodeTransport("http://worker.example.org", TOKEN)
    with pytest.raises(ValueError, match="verify"):
        NodeTransport(
            "https://worker.example.org",
            TOKEN,
            ssl_context=ssl._create_unverified_context(),
        )


def test_lost_join_ack_retries_same_membership_and_cleans_up():
    class InterruptedJoin(NodeTransport):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.scopes = []

        def exchange(self, route, request):
            result = super().exchange(route, request)
            if route == "join":
                self.scopes.append(result.scope)
                if len(self.scopes) == 1:
                    raise OSError("lost first join acknowledgement")
            return result

    with endpoint() as (registry, _, url):
        transport = InterruptedJoin(url, TOKEN)
        membership = Membership(transport)
        cleaned = Event()
        with pytest.raises(NodeUnavailable, match="stopping"):
            membership.run(
                join_request(),
                handle=lambda _: {},
                cleanup=cleaned.set,
                assignments=lambda: (),
                joined=lambda _: membership.stop.set(),
            )
        assert len(transport.scopes) == 2
        assert transport.scopes[0] == transport.scopes[1]
        assert cleaned.is_set() and not registry.inventory().nodes


def test_node_daemon_restart_reconciles_a_join_that_never_returned(
    tmp_path, monkeypatch
):
    import json
    from uuid import uuid4

    from shadowspill.orchestration.launch.admission import (
        AssignedDevice,
        NodeCapacity,
    )
    from shadowspill.orchestration.nodes import daemon as daemon_module
    from shadowspill.orchestration.nodes.daemon import NodeDaemon

    capacity = NodeCapacity(
        "worker", (AssignedDevice("test-" + uuid4().hex, "0", 100),), 1000
    )
    monkeypatch.setattr(daemon_module, "discover_capacity", lambda **_: capacity)
    with endpoint() as (registry, _, url), ThreadPoolExecutor(max_workers=1) as pool:
        first = NodeDaemon(
            url, node_id="worker", pool="lab", work_root=tmp_path, token=TOKEN
        )
        exchange = first._membership.transport.exchange

        def lost(route, request):
            response = exchange(route, request)
            if route == "join":
                raise OSError("all join acknowledgements lost")
            return response

        monkeypatch.setattr(first._membership.transport, "exchange", lost)
        with pytest.raises(OSError, match="all join"):
            first.serve()
        journal = json.loads((tmp_path / "membership.json").read_text())
        assert journal["scope"] is None and registry.inventory().nodes
        second = NodeDaemon(
            url, node_id="worker", pool="lab", work_root=tmp_path, token=TOKEN
        )
        joined = Event()
        persist = second._joined

        def recorded(value):
            persist(value)
            joined.set()

        monkeypatch.setattr(second, "_joined", recorded)
        future = pool.submit(second.serve)
        try:
            assert joined.wait(3)
            assert registry.inventory().nodes[0].id == "worker"
            recovered = json.loads((tmp_path / "membership.json").read_text())
            assert recovered["scope"] is not None
            assert recovered["incarnation"] != journal["incarnation"]
        finally:
            second.close()
        future.result(timeout=3)
        assert not registry.inventory().nodes
