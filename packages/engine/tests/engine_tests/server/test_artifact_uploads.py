"""Upload prefixes survive disconnect/restart without becoming saved artifacts."""

import hashlib
import io
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event

import pytest

from shadowspill.orchestration.launch.publication import OutputSpec
from shadowspill.orchestration.launch.staging import COPY_BYTES
from shadowspill.server import artifact_uploads
from shadowspill.server.artifacts import INDEX_BYTES, ArtifactStore
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.requests import ArtifactRef
from shadowspill_client.protocol.uploads import MAX_UPLOAD_CHUNK_BYTES


def reference(data=b"result", name="output"):
    return ArtifactRef(name, hashlib.sha256(data).hexdigest(), len(data))


@pytest.fixture
def open_store(tmp_path):
    def open_(**kwargs):
        return ArtifactStore(
            tmp_path / "artifacts",
            owner_id="owner",
            cache_root=tmp_path,
            memory=kwargs.pop("memory", BufferBudget(8 << 20, 10)),
            max_bytes=kwargs.pop("max_bytes", 16 << 20),
            max_artifacts=kwargs.pop("max_artifacts", 100),
            **kwargs,
        )

    return open_


def chunk(uploads, data, offset=0, name="output"):
    return uploads.chunk(
        name, offset, len(data), hashlib.sha256(data).hexdigest(), io.BytesIO(data)
    )


def test_partial_resume_and_publication_share_one_durable_artifact_index(open_store):
    memory = BufferBudget(8 << 20, 10)
    ref = reference()
    with open_store(memory=memory) as store:
        start = store.uploads.begin(ref)
        assert start.offset == 0 and start.state == "uploading"
        assert store.uploads.begin(ref) == start
        assert store.used == (6, 1)
        first = chunk(store.uploads, b"res")
        assert (
            first.offset == 3
            and first.prefix_sha256 == hashlib.sha256(b"res").hexdigest()
        )
        assert store.uploads.get(ref.id) == first
        assert store.get(ref.id) is None and store.page() == ()
        assert store._disk.used == (0, 0)
        # The only retained transfer memory is a bounded SHA-state cache.
        assert memory.used == (INDEX_BYTES + artifact_uploads._HASH_BYTES, 2)
    assert memory.used == (0, 0)
    with open_store(memory=memory) as store:
        assert store.uploads.begin(ref) == first
        received = chunk(store.uploads, b"ult", 3)
        assert received.offset == 6 and received.state == "uploading"
        assert store.get(ref.id) is None
        result = store.uploads.complete(ref.id)
        assert result.state == "complete" and result.expires_at is None
        assert store.used == (6, 1) and store.page() == (ref,)
        with store.open(ref) as source:
            assert source.read() == b"result"
        assert store.uploads.complete(ref.id) == result
        assert store.uploads.begin(ref) == result
        assert not list(store.uploads._directory.iterdir())
        assert memory.used == (INDEX_BYTES, 1)
    with open_store() as store:
        assert store.get(ref.id) == ref
        assert store.uploads.get(ref.id) == result
        store.uploads.abort(ref.id)  # Never deletes committed results.
        assert store.get(ref.id) == ref


def test_empty_file_publication_and_pending_completion_validation(open_store):
    with open_store() as store:
        ref = reference(b"")
        store.uploads.begin(ref)
        seen = []

        def validate(path):
            seen.append(path.read_bytes())
            raise ValueError("missing dependencies")

        with pytest.raises(ValueError, match="dependencies"):
            store.uploads.complete(ref.id, validate=validate)
        assert seen == [b""] and store.get(ref.id) is None
        assert store.uploads.complete(ref.id).state == "complete"
        pending = reference(b"pending", "another")
        store.uploads.begin(pending)
        with pytest.raises(ValueError, match="complete artifact"):
            store.uploads.complete(pending.id)


@pytest.mark.parametrize(
    "damage", ["short", "long", "checksum", "exception", "whole_checksum"]
)
def test_rejected_chunk_preserves_acknowledged_prefix_and_releases_transfer(
    open_store, damage
):
    memory = BufferBudget(8 << 20, 10)
    with open_store(memory=memory) as store:
        store.uploads.begin(reference())
        first = chunk(store.uploads, b"res")

        class Failure(io.BytesIO):
            def readinto(self, buffer):
                if self.tell():
                    raise OSError("disconnected")
                return super().readinto(buffer[:1])

        data = (
            b"ul"
            if damage == "short"
            else b"ult!"
            if damage == "long"
            else b"BAD"
            if damage == "whole_checksum"
            else b"ult"
        )
        source = Failure(data) if damage == "exception" else io.BytesIO(data)
        expected = (
            "0" * 64 if damage == "checksum" else hashlib.sha256(data[:3]).hexdigest()
        )
        with pytest.raises((ValueError, OSError)):
            store.uploads.chunk("output", 3, 3, expected, source)
        assert store.uploads.get("output") == first
        assert store._disk.used == (0, 0)
        assert not store._receiving and memory.used == (INDEX_BYTES, 1)
        assert chunk(store.uploads, b"ult", 3).offset == 6
        store.uploads.complete("output")
        with store.open(reference()) as source:
            assert source.read() == b"result"


def test_chunk_bounds_identity_conflicts_and_shared_storage_capacity(open_store):
    with open_store(max_bytes=6, max_artifacts=2) as store:
        store.uploads.begin(reference())
        with pytest.raises(ValueError, match="different content"):
            store.uploads.begin(replace(reference(), media_type="different"))
        with pytest.raises(ValueError, match="capacity"):
            store.uploads.begin(reference(b"x", "another"))
        with pytest.raises(RuntimeError, match="in progress"):
            store.put(reference(), lambda _: pytest.fail("read conflicting file"))
        with pytest.raises(RuntimeError, match="in progress"):
            store.produce(
                OutputSpec("output", 6, "application/octet-stream"),
                lambda _: pytest.fail("invoked encoder"),
            )
        for offset, size in ((1, 1), (0, 0), (0, 7), (0, MAX_UPLOAD_CHUNK_BYTES + 1)):
            with pytest.raises(ValueError):
                store.uploads.chunk("output", offset, size, "a" * 64, None)
        store.uploads.begin(reference(b"", "empty"))
        with pytest.raises(ValueError, match="capacity"):
            store.uploads.begin(reference(b"", "another"))
        store.uploads.abort("output")
        assert store.used == (0, 1)
        assert store.uploads.get("output") is None
        store.put(reference(), lambda _: io.BytesIO(b"result"))
        assert store.used == (6, 2)


def test_expiry_reclaims_only_incomplete_files_and_never_extends_on_queries(
    open_store, monkeypatch
):
    now = [1000.0]
    monkeypatch.setattr(artifact_uploads, "time", lambda: now[0])
    with open_store(max_bytes=6, upload_ttl_seconds=10) as store:
        start = store.uploads.begin(reference())
        now[0] = 1009.0
        assert store.uploads.get("output") == start
        assert store.uploads.begin(reference()) == start
        now[0] = 1010.0
        assert store.uploads.get("output") is None
        assert store.used == (0, 0) and not list(store.uploads._directory.iterdir())
        store.uploads.begin(reference())
        chunk(store.uploads, b"result")
        store.uploads.complete("output")
        now[0] = 100000.0
        store.uploads.expire()
        assert store.get("output") == reference()
    with open_store(upload_ttl_seconds=10) as store:
        store.uploads.begin(reference(b"pending", "next"))
    now[0] += 10.0
    with open_store(upload_ttl_seconds=10) as store:
        assert store.uploads.get("next") is None and store.used == (6, 1)


def test_active_producer_keeps_files_quotas_and_shared_transfer_slot(
    open_store, monkeypatch
):
    now = [1000.0]
    monkeypatch.setattr(artifact_uploads, "time", lambda: now[0])
    entered, release = Event(), Event()

    class Blocked(io.BytesIO):
        def readinto(self, buffer):
            entered.set()
            assert release.wait(5)
            return super().readinto(buffer)

    with (
        open_store(max_transfers=1, upload_ttl_seconds=10) as store,
        ThreadPoolExecutor(1) as pool,
    ):
        store.uploads.begin(reference())
        store.uploads.begin(reference(b"other", "other"))
        future = pool.submit(
            store.uploads.chunk, "output", 0, 6, reference().sha256, Blocked(b"result")
        )
        try:
            assert entered.wait(5)
            now[0] += 10
            store.uploads.expire()
            assert store.used == (6, 1)
            assert store.uploads.get("output").offset == 0
            assert store.get("output") is None
            for operation in (
                lambda: chunk(store.uploads, b"result"),
                lambda: store.uploads.abort("output"),
                lambda: store.uploads.complete("output"),
                store.close,
            ):
                with pytest.raises(RuntimeError):
                    operation()
            store.uploads.begin(reference(b"other", "other"))
            with pytest.raises(TimeoutError):
                chunk(store.uploads, b"other", name="other")
            with pytest.raises(TimeoutError):
                store.put(reference(b"next", "next"), lambda _: io.BytesIO(b"next"))
        finally:
            release.set()
        assert future.result(timeout=5).offset == 6
        assert store.uploads.complete("output").state == "complete"


@pytest.mark.parametrize("operation", ["begin", "chunk", "complete", "abort"])
@pytest.mark.parametrize("failure", ["before_commit", "after_commit", "index"])
def test_uncertain_upload_commits_reconcile_once_without_payload_replay(
    open_store, monkeypatch, operation, failure
):
    store = open_store()
    if operation != "begin":
        store.uploads.begin(reference())
    if operation == "complete":
        chunk(store.uploads, b"result")
    append = store._journal.append

    def failed_commit(entries):
        if failure == "after_commit":
            append(entries)
        raise OSError("lost durable acknowledgement")

    def failed_index(*args):
        raise OSError("derived index failed")

    if failure == "index":
        method = "remove_upload" if operation in ("abort", "complete") else "set_upload"
        monkeypatch.setattr(store._index, method, failed_index)
    else:
        monkeypatch.setattr(store._journal, "append", failed_commit)
    actions = {
        "begin": lambda: store.uploads.begin(reference()),
        "chunk": lambda: chunk(store.uploads, b"res"),
        "complete": lambda: store.uploads.complete("output"),
        "abort": lambda: store.uploads.abort("output"),
    }
    with pytest.raises(OSError):
        actions[operation]()
    assert store.used == (6, 1)
    assert not store._receiving and store._disk.used == (0, 0)
    with pytest.raises(RuntimeError, match="reconcile"):
        store.uploads.begin(reference())
    store.close()
    with open_store() as reopened:
        status = reopened.uploads.get("output")
        committed = failure != "before_commit"
        if operation == "begin":
            assert (status is not None) == committed
        elif operation == "abort":
            assert (status is None) == committed
        elif operation == "chunk":
            assert status.offset == (3 if committed else 0)
        else:
            assert (status.state == "complete") == committed
        if status is not None and status.state == "uploading":
            assert reopened.uploads._path("output").stat().st_size == status.offset
        assert reopened.used == ((6, 1) if status is not None else (0, 0))


@pytest.mark.parametrize(
    "damage", ["missing", "short", "symlink", "hardlink", "checksum"]
)
def test_retained_prefix_damage_is_not_silently_resumed(open_store, tmp_path, damage):
    with open_store() as store:
        store.uploads.begin(reference())
        chunk(store.uploads, b"res")
        path = store.uploads._path("output")
    unrelated = tmp_path / "unrelated"
    unrelated.write_bytes(b"res")
    if damage == "short":
        path.write_bytes(b"r")
    elif damage == "checksum":
        path.write_bytes(b"BAD")
    elif damage == "hardlink":
        path.unlink()
        os.link(unrelated, path)
    else:
        path.unlink()
        if damage == "symlink":
            path.symlink_to(unrelated)
    if damage == "checksum":
        with open_store() as store, pytest.raises(ValueError, match="prefix checksum"):
            chunk(store.uploads, b"ult", 3)
    else:
        with pytest.raises((OSError, ValueError)):
            open_store()
    assert unrelated.read_bytes() == b"res"


def test_transfer_scratch_is_bounded_and_sha_cache_reuses_prefix(
    open_store, monkeypatch
):
    payload = b"x" * (3 * COPY_BYTES + 17)

    class Bounded(io.BytesIO):
        def readinto(self, view):
            assert len(view) <= COPY_BYTES
            return super().readinto(view)

    with open_store(max_transfers=1) as store:
        store.uploads.begin(reference(payload))
        first = payload[: COPY_BYTES + 17]
        store.uploads.chunk(
            "output", 0, len(first), hashlib.sha256(first).hexdigest(), Bounded(first)
        )

        calls = []
        sha256 = hashlib.sha256
        remaining = payload[len(first) :]
        expected = sha256(remaining).hexdigest()

        def new_hash(*args):
            if not args:
                calls.append(True)
            return sha256(*args)

        with monkeypatch.context() as patch:
            patch.setattr(hashlib, "sha256", new_hash)
            assert store.uploads.chunk(
                "output", len(first), len(remaining), expected, Bounded(remaining)
            ).offset == len(payload)
            assert len(calls) == 1  # Only the chunk hash; the prefix uses SHA.copy.
        store.uploads.begin(reference(b"other", "other"))
        chunk(store.uploads, b"other", name="other")
        assert tuple(store.uploads._prefixes) == ("other",)
        assert store.uploads.complete("output").state == "complete"


def test_process_crash_resumes_only_its_acknowledged_prefix(open_store, tmp_path):
    child = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import hashlib, io, os, sys
from pathlib import Path
from shadowspill.server.artifacts import ArtifactStore
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.requests import ArtifactRef
root = Path(sys.argv[1])
store = ArtifactStore(root/'artifacts', owner_id='owner', cache_root=root,
                      memory=BufferBudget(8 << 20, 10), max_bytes=100,
                      max_artifacts=10)
store.uploads.begin(ArtifactRef('output', hashlib.sha256(b'result').hexdigest(), 6))
store.uploads.chunk('output', 0, 3, hashlib.sha256(b'res').hexdigest(),
                    io.BytesIO(b'res'))
class Crash(io.BytesIO):
    def readinto(self, view):
        if self.tell():
            os._exit(47)
        return super().readinto(view[:1])
store.uploads.chunk('output', 3, 3, hashlib.sha256(b'ult').hexdigest(), Crash(b'ult'))
""",
            str(tmp_path),
        ],
        capture_output=True,
        timeout=15,
    )
    assert child.returncode == 47, child.stderr
    with open_store() as store:
        assert store.uploads.get("output").offset == 3
        assert store.uploads._path("output").read_bytes() == b"res"
        assert store.get("output") is None
        chunk(store.uploads, b"ult", 3)
        store.uploads.complete("output")
        with store.open(reference()) as source:
            assert source.read() == b"result"


def test_expired_upload_does_not_prevent_reopening_with_smaller_limits(
    open_store, monkeypatch
):
    now = [1000.0]
    monkeypatch.setattr(artifact_uploads, "time", lambda: now[0])
    with open_store(upload_ttl_seconds=10) as store:
        store.uploads.begin(reference())
    now[0] += 10
    with open_store(max_bytes=1) as store:
        assert store.used == (0, 0)
