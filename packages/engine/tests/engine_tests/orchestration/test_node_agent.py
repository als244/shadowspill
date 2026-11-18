"""Atomic resource claims, idempotent stop and guarded ownership locks."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from uuid import uuid4

import pytest

from shadowspill.orchestration.launch.admission import (
    AssignedDevice,
    NodeAdmission,
    NodeCapacity,
)
from shadowspill.orchestration.nodes.agent import AgentAssignments
from shadowspill.orchestration.nodes.ownership import NodeOwnership
from shadowspill.orchestration.nodes.protocol import Command, NodeScope

SCOPE = NodeScope("controller", "membership")


def claims():
    node = NodeAdmission(
        NodeCapacity("node", [AssignedDevice("gpu", "gpu", 100)], 1000)
    )
    return node, AgentAssignments(node)


def command(sequence=1, assignment="run", operation="claim", payload=None):
    return Command(
        SCOPE,
        sequence,
        assignment,
        operation,
        {"devices": {"gpu": 90}, "host_bytes": 200} if payload is None else payload,
    )


def test_claim_stop_and_close_reject_new_ownership():
    node, owners = claims()
    owners.handle(command())
    assert node.used.claims == 1 and node.used.host_bytes == 200
    with pytest.raises(ValueError):
        owners.handle(command(sequence=2))
    stop = command(sequence=2, operation="stop", payload={})
    assert owners.handle(stop) == owners.handle(stop)
    assert node.used.claims == 0
    with pytest.raises(RuntimeError, match="stopping"):
        owners.handle(command(sequence=3))
    owners.close()
    with pytest.raises(RuntimeError, match="stopping"):
        owners.handle(command(sequence=4, assignment="other"))


def test_partial_claim_failure_has_no_leak():
    node, owners = claims()
    with pytest.raises(ValueError):
        owners.handle(command(payload={"devices": {"gpu": 101}, "host_bytes": 200}))
    assert not owners.active() and node.used.claims == 0
    owners.close()


def test_failed_child_cleanup_retains_agent_owner_for_retry(monkeypatch):
    node, owners = claims()
    owners.handle(command())
    lease = owners._leases["run"]
    close = lease.close
    monkeypatch.setattr(
        lease, "close", lambda: (_ for _ in ()).throw(RuntimeError("still live"))
    )
    with pytest.raises(BaseExceptionGroup):
        owners.close()
    assert node.used.claims == 1 and owners.active() == ("run",)
    monkeypatch.setattr(lease, "close", close)
    owners.close()
    assert node.used.claims == 0


def test_locks_cover_different_work_roots_and_retain_partial_acquisition():
    name = "test-node:" + uuid4().hex
    first, second = NodeOwnership(), NodeOwnership()
    try:
        first.acquire(name)
        with pytest.raises(RuntimeError, match="still own"):
            second.acquire(name)
        first.close()
        second.acquire(name)
    finally:
        first.close()
        second.close()


def test_heartbeat_assignment_snapshot_does_not_wait_on_child_cleanup(monkeypatch):
    _, owners = claims()
    owners.handle(command())
    entered, release = Event(), Event()
    lease = owners._leases["run"]
    close = lease.close

    def blocked_close():
        entered.set()
        assert release.wait(3)
        close()

    monkeypatch.setattr(lease, "close", blocked_close)
    with ThreadPoolExecutor(max_workers=2) as pool:
        stopping = pool.submit(owners.stop, "run")
        try:
            assert entered.wait(1)
            assert pool.submit(owners.active).result(timeout=1) == ("run",)
        finally:
            release.set()
        stopping.result(timeout=1)
    assert not owners.active()
    owners.close()
