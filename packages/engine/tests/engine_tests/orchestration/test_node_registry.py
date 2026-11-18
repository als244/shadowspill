"""Membership fencing, immutable delivery and resource cleanup boundaries."""

from dataclasses import replace
from threading import Thread
from time import monotonic, sleep

import pytest

from shadowspill.orchestration.nodes.authentication import NodeAccess, NodeTokens
from shadowspill.orchestration.nodes.protocol import (
    CODEC,
    CONTROL_BYTES,
    Acknowledged,
    Heartbeat,
    Join,
    Leave,
    NodeScope,
    Poll,
    Receipt,
    Reconcile,
)
from shadowspill.orchestration.nodes.registry import NodeRegistry, NodeUnavailable
from shadowspill.server.authentication import BearerTokens
from shadowspill_client.protocol.hardware import HardwareDevice, HardwareNode

ACCESS = NodeAccess("worker", "lab")
TOKEN = "node-test-" + "a" * 32


def join_request(incarnation="agent", previous=None):
    return Join(
        incarnation,
        HardwareNode(
            "worker",
            "lab",
            [HardwareDevice("gpu", 0, "Test GPU", 100, True)],
            1000,
            1000,
        ),
        ["shadowspill"],
        "10.0.0.2",
        previous,
    )


def registry():
    now = [0.0]
    service = NodeRegistry("controller", clock=lambda: now[0])
    joined = service.join(ACCESS, join_request())
    return service, joined, now


def test_join_identity_version_and_separate_credentials():
    service, joined, _ = registry()
    assert service.join(ACCESS, join_request()) == joined
    with pytest.raises(PermissionError):
        service.join(NodeAccess("somebody-else"), join_request())
    with pytest.raises(NodeUnavailable, match="reconciled"):
        service.join(ACCESS, join_request("other-agent"))
    with pytest.raises(ValueError, match="version"):
        replace(join_request(), version=2)
    auth = NodeTokens({TOKEN: ACCESS})
    assert auth.authorize(["Bearer " + TOKEN]) == ACCESS
    with pytest.raises(ValueError, match="distinct"):
        auth.require_separate(BearerTokens({TOKEN: "owner"}))
    with pytest.raises(PermissionError):
        auth.authorize(["Bearer client-" + "b" * 32])


def test_expiry_never_revives_or_proves_cleanup():
    service, joined, now = registry()
    service.submit(
        "worker", "run", "claim", {"devices": {"gpu": 90}, "host_bytes": 200}
    )
    assert not service.inventory().nodes[0].devices[0].available
    now[0] = 30
    assert not service.inventory().nodes
    with pytest.raises(NodeUnavailable):
        service.heartbeat("worker", Heartbeat(joined.scope, 1, ["run"]))
    with pytest.raises(NodeUnavailable):
        service.join(ACCESS, join_request("replacement"))
    with pytest.raises(NodeUnavailable):
        service.join(ACCESS, join_request())
    fresh = service.join(ACCESS, join_request("replacement", previous=joined.scope))
    assert fresh.scope != joined.scope
    with pytest.raises(NodeUnavailable):
        service.complete("worker", Receipt(joined.scope, 1))
    with pytest.raises(NodeUnavailable):
        service.leave("worker", Leave(joined.scope))


def test_reconciliation_targets_only_the_journaled_predecessor():
    service, joined, _ = registry()
    service.submit("worker", "run", "claim", {"devices": {"gpu": 90}})
    service.reconcile("worker", Reconcile("unaccepted-join"))
    assert (
        service.inventory().nodes
        and not service.inventory().nodes[0].devices[0].available
    )
    request = Reconcile("agent")
    assert service.reconcile("worker", request) == service.reconcile("worker", request)
    with pytest.raises(NodeUnavailable):
        service.heartbeat("worker", Heartbeat(joined.scope, 1, ()))
    fresh = service.join(ACCESS, join_request("replacement"))
    service.reconcile("worker", request)
    assert service.heartbeat("worker", Heartbeat(fresh.scope, 1, ()))
    assert service.inventory().nodes[0].devices[0].available


def test_deliver_retry_receipt_retirement_and_stop_failure():
    service, joined, _ = registry()
    command = service.submit("worker", "run", "claim", {"devices": {"gpu": 90}})
    assert service.poll("worker", Poll(joined.scope)) == command
    assert service.poll("worker", Poll(joined.scope)) == command
    with pytest.raises(ValueError, match="already exists"):
        service.submit("worker", "run", "claim", {"devices": {"gpu": 80}})
    receipt = Receipt(joined.scope, command.sequence, {"claimed": True})
    assert service.complete("worker", receipt) == service.complete("worker", receipt)
    with pytest.raises(ValueError, match="immutable"):
        service.complete("worker", replace(receipt, error="changed"))
    assert service.wait("worker", command) == receipt
    service.retire("worker", command)
    assert service.poll("worker", Poll(joined.scope, 1)) == Acknowledged(
        joined.scope, 1
    )
    stop = service.submit("worker", "run", "stop")
    service.complete(
        "worker", Receipt(joined.scope, stop.sequence, error="child still owns GPU")
    )
    assert not service.inventory().nodes[0].devices[0].available
    retry = service.submit("worker", "run", "stop")
    service.complete("worker", Receipt(joined.scope, retry.sequence))
    assert service.inventory().nodes[0].devices[0].available
    assert service.leave("worker", Leave(joined.scope)) == service.leave(
        "worker", Leave(joined.scope)
    )


def test_heartbeat_sequence_and_restart_fencing():
    service, joined, now = registry()
    now[0] = 10
    heartbeat = Heartbeat(joined.scope, 1, [])
    assert service.heartbeat("worker", heartbeat) == Acknowledged(joined.scope, 1)
    with pytest.raises(NodeUnavailable, match="stale"):
        service.heartbeat("worker", heartbeat)
    with pytest.raises(NodeUnavailable, match="unknown assignment"):
        service.heartbeat("worker", Heartbeat(joined.scope, 2, ["invented"]))
    now[0] = 35
    assert service.inventory().nodes
    now[0] = 40
    assert not service.inventory().nodes
    restarted = NodeRegistry("new-controller")
    fresh = restarted.join(ACCESS, join_request("new-agent", previous=joined.scope))
    assert fresh.scope.controller == "new-controller"
    with pytest.raises(NodeUnavailable):
        restarted.heartbeat("worker", heartbeat)


def test_poll_wakes_for_work_and_shutdown():
    service = NodeRegistry("controller", poll_seconds=0.3)
    joined = service.join(ACCESS, join_request())
    result = []
    thread = Thread(
        target=lambda: result.append(service.poll("worker", Poll(joined.scope)))
    )
    thread.start()
    sleep(0.01)
    start = monotonic()
    command = service.submit("worker", "run", "claim", {"devices": {"gpu": 90}})
    thread.join(timeout=1)
    assert result == [command] and monotonic() - start < 0.2
    service.close()
    with pytest.raises(NodeUnavailable):
        service.poll("worker", Poll(joined.scope, 1))


def test_control_codec_is_bounded_and_has_no_public_python_dispatch():
    service, joined, _ = registry()
    command = service.submit("worker", "run", "claim", {"devices": {"gpu": 90}})
    for record in (join_request(), joined, command, Receipt(joined.scope, 1)):
        assert CODEC.decode(CODEC.encode(record, max_bytes=CONTROL_BYTES)) == record
    with pytest.raises(ValueError, match="metadata exceeds"):
        service.submit("worker", "run", "worker", {"too_large": "x" * CONTROL_BYTES})
    with pytest.raises(ValueError, match="unknown node operation"):
        replace(command, operation="python.exec")
    with pytest.raises(NodeUnavailable):
        service.poll(
            "worker", Poll(NodeScope("wrong-controller", joined.scope.membership))
        )


def test_bound_pending_claims_and_reject_unknown_or_out_of_order_results():
    service = NodeRegistry("controller", max_commands=1)
    joined = service.join(ACCESS, join_request())
    command = service.submit("worker", "run", "claim", {"devices": {"gpu": 90}})
    with pytest.raises(TimeoutError):
        service.submit("worker", "run", "worker")
    with pytest.raises(ValueError):
        service.complete("worker", Receipt(joined.scope, 3))
    with pytest.raises(ValueError):
        service.retire("worker", command)
    service.complete("worker", Receipt(joined.scope, 1, error="claim rejected"))
    assert service.inventory().nodes[0].devices[0].available
