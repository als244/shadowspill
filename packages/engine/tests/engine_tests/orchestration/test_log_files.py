"""Log quotas outlive child termination and all active artifact readers."""

import os
import sys

import pytest

from shadowspill.orchestration.launch.admission import (
    AssignedDevice,
    NodeAdmission,
    NodeCapacity,
    ResourceRequest,
)
from shadowspill.orchestration.launch.host_arena import HostArena
from shadowspill.orchestration.launch.log_files import LogFiles
from shadowspill.orchestration.launch.staging import FileDelivery
from shadowspill_client.buffers import BufferBudget


def resources():
    node = NodeAdmission(
        NodeCapacity("node", (AssignedDevice("gpu", "0", 100),), 16 << 20)
    )
    arena = HostArena(node, 1 << 20, 100)
    lease = node.reserve(ResourceRequest({"gpu": 100}, 4 << 20))
    return node, arena, lease, BufferBudget(1 << 20, 100)


def test_real_capped_logs_remain_owned_until_borrowed_transfer_finishes(tmp_path):
    node, arena, lease, disk = resources()
    files = LogFiles(
        lease, "gpu", root=tmp_path, disk=disk, memory=arena.budget, limit_bytes=1024
    )
    code = """
import os, socket, sys
os.write(1, b'x' * (1 << 20))
with socket.socket(fileno=int(sys.argv[1])) as stream:
    stream.sendall(b'ready')
    while stream.recv(1024):
        pass
os.write(2, b'done\\n')
"""
    try:
        child = lease.launch(
            "gpu",
            lambda fd: [sys.executable, "-c", code, str(fd)],
            environment=os.environ,
            directory=files.directory,
            cwd=tmp_path,
            log_limit_bytes=files.limit_bytes,
        )
        assert child.stream.read(5) == b"ready"
        child.ready()
        for operation in (files.references, files.close):
            with pytest.raises(RuntimeError, match="termination"):
                operation()
        lease.close()
        assert child.process.logs[0].truncated
        assert node.used.host_bytes == 1 << 20
        references = files.references()
        assert references["stdout"].size_bytes == 1024
        assert references["stderr"].size_bytes == 5
        with files.open(references["stderr"]) as stream:
            assert stream.read() == b"done\n"
            with pytest.raises(RuntimeError, match="live reader"):
                files.close()
        with FileDelivery(
            list(references.values()), root=tmp_path, disk=disk, memory=arena.budget
        ) as archived:
            archived.receive(files.open)
            files.close()
            assert not files.directory.parent.exists()
            with archived.use() as resolve:
                assert resolve(references["stdout"]).read_bytes() == b"x" * 1024
                assert resolve(references["stderr"]).read_bytes() == b"done\n"
        assert disk.used == arena.budget.used == (0, 0)
    finally:
        lease.close()
        files.close()
        arena.close()
    assert node.used.claims == 0


def test_failed_deletion_retains_log_disk_and_node_memory_claims(tmp_path, monkeypatch):
    node, arena, lease, disk = resources()
    files = LogFiles(
        lease, "gpu", root=tmp_path, disk=disk, memory=arena.budget, limit_bytes=1024
    )
    lease.close()
    cleanup = files._temporary.cleanup
    before = disk.used, arena.budget.used

    def fail():
        raise OSError("storage cleanup failed")

    monkeypatch.setattr(files._temporary, "cleanup", fail)
    with pytest.raises(OSError, match="cleanup failed"):
        files.close()
    assert not files.closed and (disk.used, arena.budget.used) == before
    with pytest.raises(RuntimeError, match="live buffer owners"):
        arena.close()
    monkeypatch.setattr(files._temporary, "cleanup", cleanup)
    files.close()
    arena.close()
    assert node.used.claims == 0 and disk.used == (0, 0)


@pytest.mark.parametrize("fault", ["oversize", "symlink", "missing"])
def test_invalid_finalized_files_fail_without_releasing_their_owners(tmp_path, fault):
    _, arena, lease, disk = resources()
    files = LogFiles(
        lease, "gpu", root=tmp_path, disk=disk, memory=arena.budget, limit_bytes=10
    )
    lease.close()
    files.directory.mkdir()
    target = files.directory / "stdout.log"
    (files.directory / "stderr.log").write_bytes(b"")
    if fault == "oversize":
        target.write_bytes(b"x" * 11)
    elif fault == "symlink":
        other = tmp_path / "other"
        other.write_bytes(b"separate file")
        target.symlink_to(other)
    try:
        with pytest.raises((ValueError, OSError)):
            files.references()
        assert disk.used[1] == 1 and arena.budget.used[1] == 1
    finally:
        files.close()
        arena.close()
    if fault == "symlink":
        assert other.read_bytes() == b"separate file"


def test_reservation_failure_and_wrong_device_never_create_log_files(tmp_path):
    _, arena, lease, disk = resources()
    try:
        with pytest.raises(ValueError, match="not reserved"):
            LogFiles(lease, "other", root=tmp_path, disk=disk, memory=arena.budget)
        with pytest.raises(ValueError, match="limit"):
            LogFiles(lease, "gpu", root=tmp_path, disk=disk, memory=arena.budget)
        assert not list(tmp_path.iterdir())
        assert disk.used == arena.budget.used == (0, 0)
    finally:
        lease.close()
        arena.close()
