"""Output permits and storage quotas outlive producers and transfer consumers."""

import hashlib
import io
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from shadowspill.orchestration.launch import publication
from shadowspill.orchestration.launch.publication import (
    OutputFiles,
    OutputSpec,
    OutputWriter,
    require_output,
    write_output,
)
from shadowspill.orchestration.launch.staging import COPY_BYTES, FileDelivery
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.requests import ArtifactRef


@pytest.fixture
def storage(tmp_path):
    disk, memory = BufferBudget(1 << 20, 100), BufferBudget(4 << 20, 100)
    files = OutputFiles(root=tmp_path, disk=disk, memory=memory)
    yield files, disk, memory
    files.close()
    assert disk.used == memory.used == (0, 0)


def output(payload=b"owned output"):
    spec = OutputSpec("output.123", len(payload), "application/octet-stream")
    reference = ArtifactRef(spec.id, hashlib.sha256(payload).hexdigest(), len(payload))
    return spec, reference, payload


def produce(files, owner, reference, payload):
    def write(stream):
        for start in range(0, len(payload), COPY_BYTES):
            data = memoryview(payload)[start : start + COPY_BYTES]
            while data:
                data = data[stream.write(data) :]
        return reference

    return write_output(files.root, owner.spec, write)


def test_actual_child_writes_once_then_owned_file_transfers(tmp_path, storage):
    files, disk, memory = storage
    spec, reference, payload = output()
    owner = files.reserve(spec)
    used = disk.used[0]
    assert used > len(payload) and memory.used[0] >= COPY_BYTES
    owner.begin()
    child = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import json, sys
from pathlib import Path
from shadowspill.orchestration.launch.publication import OutputSpec, write_output
from shadowspill_client.protocol.requests import ArtifactRef
spec = OutputSpec.from_dict(json.loads(sys.argv[2]))
ref = ArtifactRef.from_dict(json.loads(sys.argv[3]))
def write(stream):
    stream.write(memoryview(b'owned output'))
    return ref
assert write_output(Path(sys.argv[1]), spec, write) == ref
""",
            str(files.root),
            json.dumps(spec.to_dict()),
            json.dumps(reference.to_dict()),
        ],
        capture_output=True,
        timeout=15,
    )
    assert child.returncode == 0, child.stderr
    with pytest.raises(RuntimeError, match="active producer"):
        owner.close()  # A file is not proof of producer completion.
    with pytest.raises(RuntimeError, match="not finished"):
        owner.verify(reference)
    owner.finished()
    assert disk.used[0] == used  # Child exit does not release file quota.
    owner.verify(reference)
    owner.verify(reference)
    with pytest.raises(FileExistsError):
        produce(files, owner, reference, payload)
    with FileDelivery([reference], root=tmp_path, disk=disk, memory=memory) as delivery:
        delivery.receive(owner.open)
        owner.close()
        assert disk.used[0] == len(payload)
        with delivery.use() as resolve:
            assert resolve(reference).read_bytes() == payload


def test_reservation_precedes_any_producer_and_failed_admission_exposes_no_permit(
    storage,
):
    files, disk, memory = storage
    spec, _, _ = output()
    before = memory.used
    with disk.reserve(disk.max_bytes):
        with pytest.raises(TimeoutError):
            files.reserve(spec)
        assert not list(files.root.iterdir()) and memory.used == before
    with memory.reserve(memory.max_bytes - before[0]):
        with pytest.raises(TimeoutError):
            files.reserve(spec)
        assert not list(files.root.iterdir()) and disk.used == (0, 0)
    with files.reserve(spec):
        with pytest.raises(ValueError, match="already reserved"):
            files.reserve(spec)
        assert require_output(files.root, spec).name.endswith(".data")
    with pytest.raises(FileNotFoundError):
        require_output(files.root, spec)


def test_verified_borrows_and_active_producer_retain_quotas(storage):
    files, disk, _ = storage
    spec, reference, payload = output()
    with files.reserve(spec) as owner:
        with pytest.raises(RuntimeError, match="no active producer"):
            owner.finished()
        with (
            pytest.raises(ValueError, match="not a verified"),
            owner.open(reference),
        ):
            pass
        owner.begin()
        with pytest.raises(RuntimeError, match="already started"):
            owner.begin()
        produce(files, owner, reference, payload)
        owner.finished()
        owner.verify(reference)
        with owner.open(reference) as stream:
            with pytest.raises(RuntimeError, match="active producer/consumer"):
                owner.close()
            with pytest.raises(RuntimeError, match="storage owners"):
                files.close()
            assert disk.used[0] > 0 and stream.read() == payload
        with pytest.raises(ValueError, match="cannot change"):
            owner.verify(replace(reference, sha256="0" * 64))
    owner.close()


@pytest.mark.parametrize("fault", ["size", "checksum", "symlink", "fifo"])
def test_output_integrity_failures_never_publish_or_release_quota(storage, fault):
    files, disk, _ = storage
    spec, reference, payload = output()
    with files.reserve(spec) as owner:
        owner.begin()
        produce(files, owner, reference, payload)
        owner.finished()
        path = require_output(files.root, spec)
        if fault == "size":
            path.write_bytes(b"short")
        elif fault == "checksum":
            path.write_bytes(b"x" * len(payload))
        else:
            path.unlink()
            if fault == "symlink":
                path.symlink_to(files.root / "elsewhere")
            else:
                publication.os.mkfifo(path)
        with pytest.raises((ValueError, OSError)):
            owner.verify(reference)
        assert owner.reference is None and disk.used[0] > 0


@pytest.mark.parametrize("fault", ["size", "type", "large", "symlink"])
def test_worker_requires_exact_bounded_regular_permit(storage, fault):
    files, _, _ = storage
    spec, _, _ = output()
    with files.reserve(spec):
        permit = next(files.root.glob("*.permit"))
        if fault == "large":
            permit.write_bytes(b"x" * (publication.PERMIT_BYTES + 1))
        elif fault == "symlink":
            permit.unlink()
            permit.symlink_to(files.root / "elsewhere")
        else:
            spec = (
                replace(spec, size_bytes=spec.size_bytes + 1)
                if fault == "size"
                else replace(spec, media_type="another/type")
            )
        with pytest.raises((ValueError, OSError)):
            require_output(files.root, spec)


@pytest.mark.parametrize("fault", ["short", "long", "reference", "write"])
def test_failed_producer_keeps_partial_storage_until_confirmed_end(storage, fault):
    files, disk, _ = storage
    spec, reference, payload = output()
    with files.reserve(spec) as owner:
        owner.begin()

        def write(stream):
            if fault == "long":
                stream.write(memoryview(payload + b"extra"))
            else:
                stream.write(memoryview(payload[:1]))
                if fault == "write":
                    raise OSError("disk full")
            return replace(reference, id="wrong") if fault == "reference" else reference

        with pytest.raises((ValueError, OSError)):
            write_output(files.root, spec, write)
        assert disk.used[0] > 0
        with pytest.raises(RuntimeError, match="active producer"):
            owner.close()
        owner.finished()  # Matching failure reply or confirmed child cleanup.


def test_failed_deletion_and_unpublished_rollback_keep_owners_reachable(
    storage, monkeypatch
):
    files, disk, memory = storage
    spec, reference, payload = output()
    owner = files.reserve(spec)
    owner.begin()
    produce(files, owner, reference, payload)
    owner.finished()
    before = disk.used, memory.used
    real_unlink = Path.unlink

    def denied(path, *args, **kwargs):
        if path.suffix == ".data":
            raise OSError("unlink failed")
        return real_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", denied)
        with pytest.raises(OSError, match="unlink failed"):
            owner.close()
        assert (disk.used, memory.used) == before and not owner.closed
    owner.close()
    real_open = Path.open

    def broken(path, *args, **kwargs):
        if path.suffix == ".permit":
            raise OSError("permit failed")
        return real_open(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", broken)
        patch.setattr(Path, "unlink", denied)
        with pytest.raises(OSError, match="unlink failed"):
            files.reserve(spec)
        assert disk.used[0] > 0
    files.close()  # Retries unpublished owners, never a live producer.


def test_bounded_writer_counts_partial_io_and_rejects_stalled_io():
    class Partial(io.BytesIO):
        def write(self, data):
            return super().write(data[:2])

    destination = Partial()
    writer = OutputWriter(destination, 3)
    assert writer.write(memoryview(b"abc")) == 2 and writer.remaining == 1
    assert writer.write(memoryview(b"c")) == 1 and writer.remaining == 0
    with pytest.raises(ValueError, match="byte bound"):
        writer.write(memoryview(b"d"))
    assert destination.getvalue() == b"abc"

    class Stalled(io.BytesIO):
        def write(self, data):
            return 0

    with pytest.raises(OSError, match="no progress"):
        OutputWriter(Stalled(), 3).write(memoryview(b"abc"))


def test_zero_byte_result_and_large_chunk_rejection(storage):
    files, _, _ = storage
    spec, reference, _ = output(b"")
    with files.reserve(spec) as owner:
        owner.begin()
        assert write_output(files.root, spec, lambda _: reference) == reference
        owner.finished()
        owner.verify(reference)
    with pytest.raises(ValueError, match="byte bound"):
        OutputWriter(io.BytesIO(), COPY_BYTES + 1).write(
            memoryview(bytes(COPY_BYTES + 1))
        )
