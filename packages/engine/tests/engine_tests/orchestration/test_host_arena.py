"""File and metadata lifetimes cannot disappear from node host accounting."""

import hashlib
import io
from concurrent.futures import ThreadPoolExecutor

import pytest

from shadowspill.orchestration.launch.admission import (
    AssignedDevice,
    NodeAdmission,
    NodeCapacity,
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
