"""Local worker files borrow committed storage instead of duplicating its bytes."""

import io
from dataclasses import replace
from pathlib import Path

import pytest

from engine_tests.server.test_artifacts import reference, store
from engine_tests.training.test_worker_node import (
    PAYLOAD,
    REFERENCE,
    configured,
    cpu_workers,
)
from shadowspill.orchestration.launch.file_grants import FileGrants, verify_file
from shadowspill.training.worker.launch import start_processes
from shadowspill_client.buffers import BufferBudget


def test_local_grants_pin_original_storage_until_last_consumer(tmp_path):
    memory = BufferBudget(4 << 20, 100)
    payload = b"retained artifact"
    ref = reference(payload)
    with (
        store(tmp_path) as artifacts,
        FileGrants(root=tmp_path, memory=memory) as grants,
    ):
        artifacts.put(ref, lambda _: io.BytesIO(payload))
        startup = grants.lend(artifacts.local_path, [ref])
        path = verify_file(grants.root, ref)
        assert path.samefile(artifacts._path(ref))
        assert artifacts.used == (len(payload), 1)
        retained = grants.retain(ref)
        assert retained is not None
        startup.close()
        assert verify_file(grants.root, ref).read_bytes() == payload
        with pytest.raises(RuntimeError, match="readers"):
            artifacts.close()
        with pytest.raises(RuntimeError, match="consumers"):
            grants.close()
        retained.close()
        assert not path.exists()
        assert artifacts._path(ref).read_bytes() == payload
    assert memory.used == (0, 0)


def test_local_grants_reject_unowned_references_and_corrupt_bytes(tmp_path):
    memory = BufferBudget(4 << 20, 100)
    ref = reference(b"correct")
    with (
        store(tmp_path) as artifacts,
        FileGrants(root=tmp_path, memory=memory) as grants,
    ):
        artifacts.put(ref, lambda _: io.BytesIO(b"correct"))
        for forged in (replace(ref, id="other-owner"), replace(ref, sha256="a" * 64)):
            with pytest.raises(KeyError, match="committed"):
                grants.lend(artifacts.local_path, [forged])
        # The child must still verify content even though storage pins the file.
        artifacts._path(ref).write_bytes(b"corrupt")
        with (
            grants.lend(artifacts.local_path, [ref]),
            pytest.raises(ValueError, match="checksum"),
        ):
            verify_file(grants.root, ref)
    assert memory.used == (0, 0)


def test_failed_local_grant_cleanup_keeps_storage_pinned(tmp_path, monkeypatch):
    memory = BufferBudget(4 << 20, 100)
    ref = reference()
    with (
        store(tmp_path) as artifacts,
        FileGrants(root=tmp_path, memory=memory) as grants,
    ):
        artifacts.put(ref, lambda _: io.BytesIO(b"result"))
        original_unlink = Path.unlink

        def denied(path, *args, **kwargs):
            if path.parent == grants.root:
                raise OSError("link removal failed")
            return original_unlink(path, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(Path, "unlink", denied)
            with pytest.raises(OSError, match="link removal failed"):
                grants.lend(artifacts.local_path, [ref, replace(ref, id="missing")])
            with pytest.raises(RuntimeError, match="readers"):
                artifacts.close()
        grants.close()
        assert artifacts._path(ref).read_bytes() == b"result"
    assert memory.used == (0, 0)


def test_local_worker_startup_fits_without_artifact_copy_and_pins_until_close(
    tmp_path, monkeypatch
):
    cpu_workers(monkeypatch)
    root = tmp_path / "workers"
    root.mkdir()
    # Exactly the two workers' log allowance; no bytes remain for a file copy.
    node, admission, arena, disk = configured(root, disk_bytes=2 * (2048 + 12288))
    try:
        with store(tmp_path) as artifacts:
            artifacts.put(REFERENCE, lambda _: io.BytesIO(PAYLOAD))
            processes = node.start(
                lambda _: pytest.fail("local startup copied an artifact stream"),
                local_source=artifacts.local_path,
                startup_timeout=5,
            )
            assert node._delivery is None
            with start_processes(processes, stop=node.stop) as group:
                assert group.valid
                assert verify_file(node.files.grants.root, REFERENCE).samefile(
                    artifacts._path(REFERENCE)
                )
                with pytest.raises(RuntimeError, match="readers"):
                    artifacts.close()
            assert all(process.child.closed for process in processes.values())
            with pytest.raises(RuntimeError, match="readers"):
                artifacts.close()
            node.close()
            assert disk.used == arena.budget.used == (0, 0)
            assert not list(root.iterdir())
    finally:
        node.close()
        arena.close()
    assert admission.used.claims == 0
