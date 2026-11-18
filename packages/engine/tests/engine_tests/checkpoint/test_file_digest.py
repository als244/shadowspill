"""Local checksum reuse requires unchanged file identity, not just a pathname."""

import hashlib
import os
from pathlib import Path

import pytest

from shadowspill.checkpoint import file_digest
from shadowspill.checkpoint.file_digest import checksum_cache, sha256_file
from shadowspill.diagnostics.progress import Progress


def track_reads(monkeypatch, path, after_read=lambda: None):
    original = Path.open
    reads = []

    class Stream:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def fileno(self):
            return self.stream.fileno()

        def seek(self, offset):
            return self.stream.seek(offset)

        def read(self, size):
            value = self.stream.read(size)
            reads.append(size)
            after_read()
            return value

    def open_file(source, *args, **kwargs):
        stream = original(source, *args, **kwargs)
        return Stream(stream) if source == path and args == ("rb",) else stream

    monkeypatch.setattr(Path, "open", open_file)
    return reads


def test_cached_identity_avoids_payload_reads_and_can_be_reverified(
    tmp_path, monkeypatch
):
    path, cache = tmp_path / "weights", tmp_path / "cache"
    payload = b"checkpoint" * 100
    path.write_bytes(payload)
    monkeypatch.setattr(
        file_digest, "time_ns", lambda: path.stat().st_ctime_ns + 3_000_000_000
    )
    expected = hashlib.sha256(payload).hexdigest()
    reads = track_reads(monkeypatch, path)
    with checksum_cache(cache):
        assert sha256_file(path, len(payload)) == expected
        assert reads
    reads.clear()
    # A new context uses the persisted entry, not an in-process memo.
    with checksum_cache(cache):
        assert sha256_file(path, len(payload)) == expected
        assert not reads
        with checksum_cache(None):
            assert sha256_file(path, len(payload)) == expected
            assert reads
        reads.clear()
        assert sha256_file(path, len(payload)) == expected
        assert not reads


@pytest.mark.parametrize("replace_inode", [False, True])
def test_same_size_and_restored_mtime_do_not_hide_a_write(
    tmp_path, monkeypatch, replace_inode
):
    path = tmp_path / "weights"
    path.write_bytes(b"before")
    with checksum_cache(tmp_path / "cache"):
        with monkeypatch.context() as initial:
            initial.setattr(
                file_digest, "time_ns", lambda: path.stat().st_ctime_ns + 3_000_000_000
            )
            old = sha256_file(path, 6)
        assert list((tmp_path / "cache").iterdir())
        facts = path.stat()
        if replace_inode:
            other = tmp_path / "replacement"
            other.write_bytes(b"after!")
            other.replace(path)
        else:
            path.write_bytes(b"after!")
        os.utime(path, ns=(facts.st_atime_ns, facts.st_mtime_ns))
        assert sha256_file(path, 6) == hashlib.sha256(b"after!").hexdigest() != old


def test_changed_during_read_is_rejected_without_publishing_a_digest(
    tmp_path, monkeypatch
):
    path, cache = tmp_path / "weights", tmp_path / "cache"
    path.write_bytes(b"before")
    changed = False

    def change():
        nonlocal changed
        if not changed:
            changed = True
            path.write_bytes(b"after!")

    track_reads(monkeypatch, path, change)
    with (
        checksum_cache(cache),
        pytest.raises(ValueError, match="changed while pinning"),
    ):
        sha256_file(path, 6)
    assert not cache.exists()


def test_invalid_or_unwritable_cache_falls_back_to_a_content_read(
    tmp_path, monkeypatch
):
    path, cache = tmp_path / "weights", tmp_path / "cache"
    path.write_bytes(b"original")
    monkeypatch.setattr(
        file_digest, "time_ns", lambda: path.stat().st_ctime_ns + 3_000_000_000
    )
    with checksum_cache(cache):
        expected = sha256_file(path, 8)
        next(cache.iterdir()).write_bytes(b"broken cache")
        assert sha256_file(path, 8) == expected
    occupied = tmp_path / "not-a-directory"
    occupied.write_bytes(b"occupied")
    with checksum_cache(occupied):
        assert sha256_file(path, 8) == expected


def test_cached_source_still_requires_its_declared_size(tmp_path):
    path = tmp_path / "weights"
    path.write_bytes(b"original")
    with checksum_cache(tmp_path / "cache"):
        sha256_file(path, 8)
        with pytest.raises(ValueError, match="size/type"):
            sha256_file(path, 7)


def test_progress_reports_payload_scan_but_not_cached_reads(tmp_path, monkeypatch):
    path = tmp_path / "model-00001-of-00002.safetensors"
    path.write_bytes(b"weights")
    monkeypatch.setattr(
        file_digest, "time_ns", lambda: path.stat().st_ctime_ns + 3_000_000_000
    )
    events = []
    with Progress(events.append).activate(), checksum_cache(tmp_path / "cache"):
        sha256_file(path, 7)
        assert len(events) == 2
        assert "→ Fingerprint model-00001-of-00002.safetensors" in events[0]
        assert "✓ Fingerprint model-00001-of-00002.safetensors" in events[1]
        assert " — " in events[1]
        events.clear()
        sha256_file(path, 7)
        assert events == []
