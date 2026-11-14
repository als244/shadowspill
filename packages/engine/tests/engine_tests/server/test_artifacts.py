"""Committed binary results survive producers, lost replies and service restart."""

import hashlib
import io
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event

import pytest

from shadowspill.orchestration.launch.staging import COPY_BYTES, FileDelivery
from shadowspill.server.artifacts import INDEX_BYTES, ArtifactStore
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.requests import ArtifactRef


def reference(data=b"result", name="output"):
    return ArtifactRef(name, hashlib.sha256(data).hexdigest(), len(data))


def store(root, *, memory=None, owner="owner", max_bytes=16 << 20, max_artifacts=100):
    return ArtifactStore(
        root / "artifacts",
        owner_id=owner,
        cache_root=root,
        memory=memory or BufferBudget(8 << 20, 10),
        max_bytes=max_bytes,
        max_artifacts=max_artifacts,
    )


def never_open(_):
    raise AssertionError("duplicate or rejected publication opened a source")


def test_durable_result_and_retry_survive_reopen_in_framework_free_child(tmp_path):
    memory = BufferBudget(8 << 20, 10)
    ref = reference()
    with store(tmp_path, memory=memory) as artifacts:
        assert artifacts.put(ref, lambda _: io.BytesIO(b"result")) == ref
        assert artifacts.put(ref, never_open) == ref
        assert artifacts.used == (6, 1) and artifacts.page() == (ref,)
        assert artifacts._disk.used == (0, 0)
        assert memory.used == (INDEX_BYTES, 1)
    assert memory.used == (0, 0)
    assert not list(tmp_path.glob("artifact-index-*"))
    child = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import importlib.abc, sys
from pathlib import Path
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, *args):
        if name.split('.')[0] in {'torch', 'triton', 'pynvml', 'shadowspill_workloads'}:
            raise AssertionError('framework in artifact service: ' + name)
sys.meta_path.insert(0, Block())
from shadowspill.server.artifacts import ArtifactStore
from shadowspill_client.buffers import BufferBudget
root = Path(sys.argv[1])
with ArtifactStore(root/'artifacts', owner_id='owner', cache_root=root,
                   memory=BufferBudget(8 << 20, 10), max_bytes=100,
                   max_artifacts=10) as artifacts:
    ref = artifacts.get('output')
    assert ref is not None and artifacts.used == (6, 1)
    with artifacts.open(ref) as stream:
        sys.stdout.buffer.write(stream.read())
""",
            str(tmp_path),
        ],
        capture_output=True,
        timeout=15,
    )
    assert child.returncode == 0, child.stderr
    assert child.stdout == b"result"


def test_stream_is_bounded_and_pagination_uses_saved_metadata(tmp_path):
    data = b"x" * (5 * COPY_BYTES + 13)
    reads = []

    class Source(io.BytesIO):
        def read(self, size=-1):
            assert 0 < size <= COPY_BYTES
            reads.append(size)
            return super().read(size)

    with store(tmp_path) as artifacts:
        for name in ("c", "a", "b"):
            artifacts.put(reference(data, name), lambda _: Source(data))
        assert len(reads) == 21
        assert [r.id for r in artifacts.page(limit=2)] == ["a", "b"]
        assert [r.id for r in artifacts.page(after="b")] == ["c"]
        with pytest.raises(ValueError, match=r"1\.\.128"):
            artifacts.page(limit=129)
        with artifacts.open(reference(data, "a")) as stream:
            assert stream.read() == data
            with pytest.raises(RuntimeError, match="readers"):
                artifacts.close()


@pytest.mark.parametrize("data", [b"resul", b"result!", b"RESULT"])
def test_bad_payload_never_publishes_and_releases_copy_capacity(tmp_path, data):
    memory = BufferBudget(8 << 20, 10)
    with store(tmp_path, memory=memory) as artifacts:
        with pytest.raises(ValueError):
            artifacts.put(reference(), lambda _: io.BytesIO(data))
        assert artifacts.get("output") is None and artifacts.used == (0, 0)
        assert artifacts._disk.used == (0, 0)
        assert memory.used == (INDEX_BYTES, 1)
        assert not list(artifacts._scratch.iterdir())
        assert (
            artifacts.put(reference(), lambda _: io.BytesIO(b"result")) == reference()
        )


def test_capacity_and_identity_conflicts_fail_before_opening_a_source(tmp_path):
    with store(tmp_path, max_bytes=6, max_artifacts=2) as artifacts:
        artifacts.put(reference(), lambda _: io.BytesIO(b"result"))
        with pytest.raises(ValueError, match="different content"):
            artifacts.put(replace(reference(), media_type="other"), never_open)
        with pytest.raises(ValueError, match="capacity"):
            artifacts.put(reference(b"x", "next"), never_open)
        empty = reference(b"", "empty")
        artifacts.put(empty, lambda _: io.BytesIO())
        with pytest.raises(ValueError, match="capacity"):
            artifacts.put(replace(empty, id="another"), never_open)
    with store(tmp_path, max_bytes=6, max_artifacts=2) as reopened:
        assert reopened.used == (6, 2)
        assert reopened.put(reference(), never_open) == reference()
    with pytest.raises(ValueError, match="retained"):
        store(tmp_path, max_bytes=5)


def test_owner_scope_and_exclusive_journal_survive_reopen(tmp_path):
    with store(tmp_path) as artifacts:
        artifacts.put(reference(), lambda _: io.BytesIO(b"result"))
        with pytest.raises(RuntimeError, match="active writer"):
            store(tmp_path)
        with (
            pytest.raises(KeyError, match="owner"),
            artifacts.open(replace(reference(), sha256="1" * 64)),
        ):
            pass
    memory = BufferBudget(8 << 20, 10)
    with pytest.raises(ValueError, match="another owner"):
        store(tmp_path, owner="different", memory=memory)
    assert memory.used == (0, 0)
    with store(tmp_path) as reopened:
        assert reopened.get("output") == reference()


@pytest.mark.parametrize("failure", ["before_commit", "after_commit", "index"])
def test_uncertain_publication_reconciles_files_against_durable_journal(
    tmp_path, monkeypatch, failure
):
    artifacts = store(tmp_path)
    append = artifacts._journal.append

    def fail_commit(entries):
        if failure == "after_commit":
            append(entries)
        raise OSError("publication acknowledgement failed")

    def fail_index(_):
        raise OSError("disposable index failed")

    if failure == "index":
        monkeypatch.setattr(artifacts._index, "insert", fail_index)
    else:
        monkeypatch.setattr(artifacts._journal, "append", fail_commit)
    with pytest.raises(OSError):
        artifacts.put(reference(), lambda _: io.BytesIO(b"result"))
    assert artifacts.used == (6, 1)  # Uncertain bytes remain charged until recovery.
    with pytest.raises(RuntimeError, match="reconcile"):
        artifacts.put(reference(), never_open)
    artifacts.close()
    with store(tmp_path) as reopened:
        if failure == "before_commit":
            assert reopened.get("output") is None and reopened.used == (0, 0)
            assert not list(reopened._data.iterdir())
        else:
            assert reopened.put(reference(), never_open) == reference()
            with reopened.open(reference()) as stream:
                assert stream.read() == b"result"


def test_active_copy_is_not_visible_and_shutdown_keeps_its_ownership(tmp_path):
    entered, release = Event(), Event()
    memory = BufferBudget(8 << 20, 10)

    class Source(io.BytesIO):
        def read(self, size=-1):
            entered.set()
            assert release.wait(5)
            return super().read(size)

    with store(tmp_path, memory=memory) as artifacts, ThreadPoolExecutor(1) as pool:
        future = pool.submit(artifacts.put, reference(), lambda _: Source(b"result"))
        try:
            assert entered.wait(5)
            assert artifacts.get("output") is None and not artifacts.page()
            assert artifacts.used == (6, 1) and memory.used[0] > INDEX_BYTES
            with pytest.raises(RuntimeError, match="in progress"):
                artifacts.put(reference(), never_open)
            with pytest.raises(RuntimeError, match="producers"):
                artifacts.close()
        finally:
            release.set()
        assert future.result(timeout=5) == reference()
    assert memory.used == (0, 0)


def test_failed_temporary_cleanup_retains_owner_and_does_not_erase_commit(
    tmp_path, monkeypatch
):
    memory = BufferBudget(8 << 20, 10)
    artifacts = store(tmp_path, memory=memory)
    close = FileDelivery.close

    def denied(_):
        raise PermissionError("pending file is busy")

    monkeypatch.setattr(FileDelivery, "close", denied)
    with pytest.raises(PermissionError):
        artifacts.put(reference(), lambda _: io.BytesIO(b"result"))
    assert artifacts._pending and artifacts._disk.used == (6, 1)
    assert memory.used[0] > INDEX_BYTES
    with pytest.raises(PermissionError):
        artifacts.close()
    monkeypatch.setattr(FileDelivery, "close", close)
    artifacts.close()
    assert memory.used == (0, 0)
    with store(tmp_path) as reopened:
        assert reopened.get("output") == reference()


@pytest.mark.parametrize("damage", ["missing", "truncated", "symlink"])
def test_missing_or_nonregular_committed_payload_is_a_recovery_error(tmp_path, damage):
    with store(tmp_path) as artifacts:
        artifacts.put(reference(), lambda _: io.BytesIO(b"result"))
        path = artifacts._path(reference())
    if damage == "truncated":
        path.write_bytes(b"bad")
    else:
        path.unlink()
        if damage == "symlink":
            target = tmp_path / "unrelated"
            target.write_bytes(b"result")
            path.symlink_to(target)
    with pytest.raises((OSError, ValueError)):
        store(tmp_path)
    if damage == "symlink":
        assert target.read_bytes() == b"result"


def test_crashed_copy_partials_are_private_and_never_recovered_as_results(tmp_path):
    child = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import io, os, sys
from pathlib import Path
from shadowspill.server.artifacts import ArtifactStore
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.requests import ArtifactRef
class Crash(io.BytesIO):
    def read(self, size=-1):
        os._exit(47)
root = Path(sys.argv[1])
artifacts = ArtifactStore(root/'artifacts', owner_id='owner', cache_root=root,
                         memory=BufferBudget(8 << 20, 10), max_bytes=100,
                         max_artifacts=10)
artifacts.put(ArtifactRef('output', '0'*64, 6), lambda _: Crash(b'result'))
""",
            str(tmp_path),
        ],
        timeout=15,
        capture_output=True,
    )
    assert child.returncode == 47, child.stderr
    unrelated = tmp_path / "unrelated"
    unrelated.write_bytes(b"preserve me")
    with store(tmp_path) as artifacts:
        assert not artifacts.page() and artifacts.used == (0, 0)
        assert not list(artifacts._scratch.iterdir())
        assert unrelated.read_bytes() == b"preserve me"
