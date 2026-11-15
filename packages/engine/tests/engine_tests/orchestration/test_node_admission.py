"""Whole-node claims precede launch and survive incomplete process cleanup."""

import os
import sys
from dataclasses import replace
from threading import Event, Thread

import pytest

from shadowspill.orchestration.launch import admission
from shadowspill.orchestration.launch.admission import (
    AssignedDevice,
    NodeAdmission,
    NodeCapacity,
    ResourceRequest,
    ResourcesUnavailable,
)


def capacity():
    return NodeCapacity(
        "assigned-node",
        [AssignedDevice("a", "GPU-assigned-a", 100), AssignedDevice("b", "3", 200)],
        host_bytes=120,
    )


def request(device="a", *, host=70):
    return ResourceRequest({device: 50}, host)


def test_claim_is_atomic_across_host_memory_and_exclusive_assigned_devices():
    node = NodeAdmission(capacity())
    with node.reserve(request()) as first:
        before = node.used
        for blocked in (
            request("b", host=60),
            request(host=1),
        ):
            with pytest.raises(ResourcesUnavailable):
                node.reserve(blocked)
            assert node.used == before
        with node.reserve(request("b", host=40)):
            assert node.used.host_bytes == 110
            assert node.used.devices == frozenset({"a", "b"})
        assert node.used == before
        assert first.request.devices == {"a": 50}
        with pytest.raises(TypeError):
            first.request.devices["b"] = 50
    assert node.used.host_bytes == node.used.claims == 0
    assert not node.used.devices


def test_host_preparation_masks_gpus_and_preserves_an_unrelated_device_claim(tmp_path):
    node = NodeAdmission(capacity())
    with node.reserve(request(host=0)):
        with node.reserve(ResourceRequest({}, 70)) as lease:
            script = (
                "import os,socket,sys; "
                "socket.socket(fileno=int(sys.argv[1])).sendall("
                "os.environ['CUDA_VISIBLE_DEVICES'].encode() or b'<empty>')"
            )
            child = lease.launch_host(
                lambda fd: [sys.executable, "-c", script, str(fd)],
                environment={**os.environ, "CUDA_VISIBLE_DEVICES": "0,1"},
                directory=tmp_path / "logs",
                cwd=tmp_path,
            )
            assert child.stream.read(7) == b"<empty>"
            assert child.process.wait(5) == 0
            assert node.used.devices == {"a"}
            with pytest.raises(ValueError, match="unused host-only"):
                lease.launch_host(
                    lambda fd: [], environment={}, directory=tmp_path, cwd=tmp_path
                )
        assert node.used.devices == {"a"} and node.used.host_bytes == 0
    assert node.used.claims == 0


@pytest.mark.parametrize(
    "invalid",
    [
        ResourceRequest({"other": 1}, 0),
        ResourceRequest({"a": 101}, 0),
        ResourceRequest({}, 121),
    ],
)
def test_impossible_requests_fail_even_with_unbounded_resource_wait(invalid):
    node = NodeAdmission(capacity())
    with pytest.raises(ValueError, match="cannot fit"):
        node.reserve(invalid, wait=True)
    assert node.used.claims == 0


@pytest.mark.parametrize("shutdown", [False, True])
def test_waiters_wake_on_release_or_node_shutdown(shutdown):
    node = NodeAdmission(capacity())
    first = node.reserve(request())
    entered, done = Event(), Event()
    result = []

    def acquire():
        entered.set()
        try:
            with node.reserve(request(), wait=True, timeout=2):
                result.append("admitted")
        except RuntimeError as error:
            result.append(str(error))
        finally:
            done.set()

    thread = Thread(target=acquire)
    thread.start()
    assert entered.wait(1) and not done.wait(0.02)
    if shutdown:
        node.close()
        assert node.used.claims == 1
    else:
        first.close()
    thread.join(timeout=3)
    assert not thread.is_alive()
    assert result == ["node admission is closed" if shutdown else "admitted"]
    first.close()
    assert node.used.claims == 0


def test_timeout_and_metadata_claim_limit_do_not_partially_reserve():
    node = NodeAdmission(capacity(), max_claims=1)
    with node.reserve(ResourceRequest({}, 0)):
        with pytest.raises(ResourcesUnavailable):
            node.reserve(request())
        with pytest.raises(TimeoutError):
            node.reserve(request(), wait=True, timeout=0)
        assert node.used.host_bytes == 0
        assert node.used.claims == 1
    with pytest.raises(ValueError, match="wait=True"):
        node.reserve(request(), timeout=1)


def test_launch_is_once_per_reserved_slot_with_visibility_set_before_import(tmp_path):
    node = NodeAdmission(capacity())
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "unassigned,other"}
    code = """
import os, socket, sys
with socket.socket(fileno=int(sys.argv[1])) as stream:
    stream.sendall(os.environ['CUDA_VISIBLE_DEVICES'].encode())
    while stream.recv(1024):
        pass
print('orderly closed')
"""
    with node.reserve(request()) as lease:
        options = dict(
            command=lambda fd: [sys.executable, "-c", code, str(fd)],
            environment=environment,
            directory=tmp_path / "logs",
            cwd=tmp_path,
            log_limit_bytes=1024,
            startup_timeout=5,
        )
        with pytest.raises(ValueError, match="unreserved"):
            lease.launch("b", **options)
        child = lease.launch("a", **options)
        assert child.stream.read(100) == b"GPU-assigned-a"
        child.ready()
        assert environment["CUDA_VISIBLE_DEVICES"] == "unassigned,other"
        with pytest.raises(ValueError, match="already launched"):
            lease.launch("a", **options)
        assert node.used.claims == 1
    assert child.closed and child.process.returncode == 0
    assert (tmp_path / "logs/stdout.log").read_text() == "orderly closed\n"
    assert node.used.claims == 0
    with pytest.raises(RuntimeError, match="closed"):
        lease.launch("a", **options)


@pytest.mark.parametrize("failure", ["raises", "unconfirmed"])
def test_cleanup_disconnects_all_peers_first_and_holds_claim_on_failure(
    tmp_path, monkeypatch, failure
):
    node = NodeAdmission(capacity())
    lease = node.reserve(ResourceRequest({"a": 50, "b": 50}, 100))
    events, peers = [], []

    class Child:
        def __init__(self, command, **kwargs):
            self.rank = len(peers)
            self.closed = False
            self.fail = self.rank == 0
            peers.append(self)

        def disconnect(self):
            events.append((self.rank, "disconnect"))

        def close(self):
            events.append((self.rank, "close"))
            if self.fail:
                if failure == "unconfirmed":
                    return
                raise OSError("child termination is unconfirmed")
            self.closed = True

    monkeypatch.setattr(admission, "ConnectedProcess", Child)
    for name in ("a", "b"):
        lease.launch(
            name, lambda _: [], environment={}, directory=tmp_path, cwd=tmp_path
        )
    with pytest.raises(ExceptionGroup, match="did not finish cleanup"):
        lease.close()
    assert events == [(0, "disconnect"), (1, "disconnect"), (0, "close"), (1, "close")]
    assert node.used.claims == 1 and node.used.devices == frozenset({"a", "b"})
    assert peers[1].closed and not lease.closed
    with pytest.raises(ResourcesUnavailable):
        node.reserve(request())
    peers[0].fail = False
    lease.close()
    assert node.used.claims == 0 and lease.closed
    lease.close()


def test_failed_launch_keeps_claim_and_releases_it_on_lease_close(tmp_path):
    node = NodeAdmission(capacity())
    with node.reserve(request()) as lease:
        with pytest.raises(FileNotFoundError):
            lease.launch(
                "a",
                lambda _: ["/no/such/program"],
                environment={},
                directory=tmp_path / "logs",
                cwd=tmp_path,
            )
        assert node.used.claims == 1
    assert node.used.claims == 0


def test_capacity_rejects_duplicate_aliases_and_invalid_visibility():
    original = capacity()
    for field in ("id", "visibility"):
        with pytest.raises(ValueError, match="unique"):
            replace(
                original,
                devices=[
                    original.devices[0],
                    replace(
                        original.devices[1],
                        **{field: getattr(original.devices[0], field)},
                    ),
                ],
            )
    for value in ("", "-1", "0,1", "../device"):
        with pytest.raises(ValueError):
            AssignedDevice("a", value, 100)
