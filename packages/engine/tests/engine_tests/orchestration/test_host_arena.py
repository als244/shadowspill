"""File and metadata lifetimes cannot disappear from node host accounting."""

import hashlib
import io
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from shadowspill.orchestration.launch.admission import (
    AssignedDevice,
    NodeAdmission,
    NodeCapacity,
    ReservationCancelled,
    ResourceRequest,
    ResourcesUnavailable,
)
from shadowspill.orchestration.launch.host_arena import HostArena
from shadowspill.orchestration.launch.staging import FileDelivery
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.requests import ArtifactRef


def node():
    return NodeAdmission(
        NodeCapacity("local", (AssignedDevice("gpu", "0", 100),), 1000)
    )


def test_complete_arena_capacity_is_charged_independently_of_workers_and_files():
    admission = node()
    with HostArena(admission, 300, 3) as artifacts:
        assert admission.used.host_bytes == 300 and artifacts.budget.used == (0, 0)
        artifact = artifacts.budget.reserve(100)
        with admission.reserve(ResourceRequest({"gpu": 80}, 600)):
            assert admission.used.host_bytes == 900
            with pytest.raises(ResourcesUnavailable):
                HostArena(admission, 101, 1)
        assert admission.used.host_bytes == 300 and artifacts.budget.used == (100, 1)
        artifact.close()
    assert artifacts.closed and admission.used.host_bytes == admission.used.claims == 0
    artifacts.close()


def test_failed_cleanup_retains_arena_claim_and_wakes_waiting_producers():
    admission = node()
    arena = HostArena(admission, 300, 1)
    owner = arena.budget.reserve(300)
    with ThreadPoolExecutor(1) as pool:
        waiting = pool.submit(arena.budget.reserve, 1)
        with pytest.raises(RuntimeError, match="live buffer owners"):
            arena.close()
        with pytest.raises(RuntimeError, match="staging is closed"):
            waiting.result(timeout=5)
    assert not arena.closed and admission.used.host_bytes == 300
    with pytest.raises(RuntimeError, match="closing"):
        arena.__enter__()
    with pytest.raises(RuntimeError, match="staging is closed"):
        arena.budget.reserve(0)
    owner.close()
    arena.close()
    assert arena.closed and admission.used.claims == 0


def test_delivered_file_retains_its_node_charge_after_worker_release(tmp_path):
    admission = NodeAdmission(
        NodeCapacity("local", (AssignedDevice("gpu", "0", 100),), 16 << 20)
    )
    payload = b"owned binary bytes"
    reference = ArtifactRef("input", hashlib.sha256(payload).hexdigest(), len(payload))
    arena, disk = HostArena(admission, 2 << 20, 100), BufferBudget(1 << 20, 100)
    delivery = FileDelivery([reference], root=tmp_path, disk=disk, memory=arena.budget)
    delivery.receive(lambda _: io.BytesIO(payload))
    with admission.reserve(ResourceRequest({"gpu": 100}, 4 << 20)):
        assert admission.used.host_bytes == 6 << 20
    assert admission.used.host_bytes == 2 << 20
    with pytest.raises(RuntimeError, match="live buffer owners"):
        arena.close()
    assert admission.used.claims == 1
    delivery.close()
    arena.close()
    assert arena.closed and admission.used.claims == 0 and disk.used == (0, 0)


def test_zero_byte_owner_still_retains_its_arena_and_impossible_arena_does_not_wait():
    admission = node()
    with pytest.raises(ValueError, match="cannot fit"):
        HostArena(admission, 1001, 1, wait=True)
    assert admission.used.claims == 0
    arena = HostArena(admission, 100, 1)
    owner = arena.budget.reserve(0)
    with pytest.raises(RuntimeError, match="live buffer owners"):
        arena.close()
    assert admission.used.host_bytes == 100
    owner.close()
    arena.close()


def test_partitions_share_one_atomic_claim_and_close_independently():
    admission = node()
    with HostArena(admission, 900, 3) as arena:
        inputs = arena.partition(600, 2)
        results = arena.partition(300, 10)
        assert admission.used.claims == 1 and arena.budget.used == (900, 2)
        with pytest.raises(TimeoutError):
            arena.partition(1, 1)
        active = inputs.budget.reserve(600)
        with pytest.raises(RuntimeError, match="live buffer owners"):
            inputs.close()
        assert not inputs.closed and arena.budget.used == (900, 2)
        with pytest.raises(RuntimeError, match="closing"):
            inputs.__enter__()
        with results.budget.reserve(300):
            assert inputs.budget.used == (600, 1)
        active.close()
        inputs.close()
        assert arena.budget.used == (300, 1)
        with results.budget.reserve(100):
            pass
        results.close()
        assert arena.budget.used == (0, 0)
        assert admission.used.host_bytes == 900
    assert admission.used.claims == 0


def test_parent_retains_its_full_claim_until_the_last_partition_is_closed():
    admission = node()
    arena = HostArena(admission, 900, 2)
    part = arena.partition(300, 1)
    owner = part.budget.reserve(0)
    with pytest.raises(RuntimeError, match="live buffer owners"):
        arena.close()
    with pytest.raises(RuntimeError, match="closing"):
        arena.partition(1, 1)
    with pytest.raises(RuntimeError, match="live buffer owners"):
        part.close()
    assert admission.used.host_bytes == 900
    owner.close()
    part.close()
    arena.close()
    assert admission.used.claims == 0


def test_cancelled_whole_arena_wait_never_acquires_a_partial_reservation(monkeypatch):
    admission = node()
    occupied = admission.reserve(ResourceRequest({}, 900))
    entered, cancelled = Event(), Event()
    reserve = admission.reserve

    def waiting(*args, **kwargs):
        entered.set()
        return reserve(*args, **kwargs)

    monkeypatch.setattr(admission, "reserve", waiting)
    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(
            HostArena, admission, 500, 4, wait=True, cancelled=cancelled
        )
        assert entered.wait(5)
        assert admission.used.host_bytes == 900 and admission.used.claims == 1
        cancelled.set()
        with pytest.raises(ReservationCancelled):
            pending.result(timeout=5)
    assert not occupied.closed
    occupied.close()
    assert admission.used.claims == 0


def test_child_process_demand_is_reported_fully_but_charged_to_parent_once():
    admission = node()
    with HostArena(admission, 900, 2) as arena:
        host = arena.partition(700, 1)
        scratch = arena.partition(200, 1)
        child = admission.reserve(ResourceRequest({"gpu": 100}, 600), host=host)
        assert child.request.host_bytes == 600 and admission.used.host_bytes == 900
        assert admission.used.claims == 2 and host.budget.used == (600, 1)
        with pytest.raises(RuntimeError, match="live buffer owners"):
            host.close()
        with scratch.budget.reserve(200):
            child.close()
            assert host.budget.used == (0, 0) and admission.used.host_bytes == 900
        host.close()
        scratch.close()
    assert admission.used.claims == 0


def test_preclaimed_memory_cannot_cross_nodes_or_leak_after_failed_device_admission():
    admission, other = node(), node()
    with HostArena(admission, 900, 1) as arena, arena.partition(800, 1) as host:
        with pytest.raises(ValueError, match="another node"):
            other.reserve(ResourceRequest({"gpu": 100}, 600), host=host)
        assert host.budget.used == (0, 0) and other.used.claims == 0
        with (
            admission.reserve(ResourceRequest({"gpu": 100}, 0)),
            pytest.raises(ResourcesUnavailable),
        ):
            admission.reserve(ResourceRequest({"gpu": 100}, 600), host=host)
        assert host.budget.used == (0, 0)


def test_failed_child_exit_retains_its_preclaimed_slice(monkeypatch):
    admission = node()
    with HostArena(admission, 900, 1) as arena, arena.partition(800, 1) as host:
        child = admission.reserve(ResourceRequest({"gpu": 100}, 600), host=host)

        class FailedChild:
            closed = False

            def disconnect(self):
                pass

            def close(self):
                raise OSError("exit not confirmed")

        failed = FailedChild()
        child._children["gpu"] = failed
        with pytest.raises(ExceptionGroup):
            child.close()
        assert host.budget.used == (600, 1) and admission.used.devices == {"gpu"}
        monkeypatch.setattr(failed, "close", lambda: setattr(failed, "closed", True))
        child.close()
        assert host.budget.used == (0, 0) and not admission.used.devices
