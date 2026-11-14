"""File access across processes must retain the delivery's storage ownership."""

import hashlib
import io
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from shadowspill.orchestration.launch import file_grants
from shadowspill.orchestration.launch.file_grants import FileGrants, resolve_file
from shadowspill.orchestration.launch.staging import FileDelivery
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.requests import ArtifactRef


@pytest.fixture
def delivered(tmp_path):
    payload = b"immutable input bytes"
    ref = ArtifactRef("input", hashlib.sha256(payload).hexdigest(), len(payload))
    disk, memory = (
        BufferBudget(10000, max_requests=100),
        BufferBudget(4 << 20, max_requests=100),
    )
    delivery = FileDelivery([ref], root=tmp_path, disk=disk, memory=memory)
    delivery.receive(lambda _: io.BytesIO(payload))
    yield delivery, ref, disk, memory, payload
    delivery.close()
    assert disk.used[0] == memory.used[0] == 0


def test_actual_child_resolves_only_the_granted_opaque_reference(tmp_path, delivered):
    delivery, ref, disk, memory, payload = delivered
    with FileGrants(root=tmp_path, memory=memory) as grants:
        with grants.lend(delivery, [ref]):
            child = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    """
import json, sys
from pathlib import Path
from shadowspill.orchestration.launch.file_grants import resolve_file
from shadowspill_client.protocol.requests import ArtifactRef
path = resolve_file(Path(sys.argv[1]), ArtifactRef.from_dict(json.loads(sys.argv[2])))
sys.stdout.buffer.write(path.read_bytes())
""",
                    str(grants.root),
                    json.dumps(ref.to_dict()),
                ],
                capture_output=True,
                timeout=15,
            )
            assert child.returncode == 0, child.stderr
            assert child.stdout == payload and disk.used[0] == len(payload)
            with pytest.raises(RuntimeError, match="consumers"):
                delivery.close()
            with pytest.raises(RuntimeError, match="consumers"):
                grants.close()
            for changed in (replace(ref, id="other"), replace(ref, media_type="other")):
                with pytest.raises(FileNotFoundError):
                    resolve_file(grants.root, changed)
        with pytest.raises(FileNotFoundError):
            resolve_file(grants.root, ref)


def test_shared_grants_retain_one_delivery_until_its_last_consumer(tmp_path, delivered):
    delivery, ref, _, memory, payload = delivered
    with FileGrants(root=tmp_path, memory=memory) as grants:
        first, second = grants.lend(delivery, [ref]), grants.lend(delivery, [ref])
        assert len(list(grants.root.iterdir())) == 1
        first.close()
        assert resolve_file(grants.root, ref).read_bytes() == payload
        with pytest.raises(RuntimeError, match="consumers"):
            delivery.close()
        second.close()
        second.close()
        assert not list(grants.root.iterdir())


def test_identical_ref_from_another_delivery_cannot_transfer_disk_ownership(
    tmp_path, delivered
):
    delivery, ref, disk, memory, payload = delivered
    with (
        FileDelivery([ref], root=tmp_path, disk=disk, memory=memory) as other,
        FileGrants(root=tmp_path, memory=memory) as grants,
    ):
        other.receive(lambda _: io.BytesIO(payload))
        with grants.lend(delivery, [ref]):
            before = memory.used[0]
            with pytest.raises(ValueError, match="another delivery"):
                grants.lend(other, [ref])
            assert memory.used[0] == before
            other.close()
            assert disk.used[0] == len(payload)
            assert resolve_file(grants.root, ref).read_bytes() == payload


@pytest.mark.parametrize("failure", ["reference", "duplicate", "link", "space"])
def test_partial_grants_are_removed_and_borrowed_quotas_retained_correctly(
    tmp_path, delivered, monkeypatch, failure
):
    delivery, ref, _, memory, _ = delivered
    with FileGrants(root=tmp_path, memory=memory) as grants:
        before = memory.used[0]
        refs = (
            [ref, replace(ref, id="missing")] if failure == "reference" else [ref, ref]
        )
        if failure == "link":

            def denied(*args):
                raise OSError("different filesystem")

            monkeypatch.setattr(file_grants.os, "link", denied)
        extra = None
        if failure == "space":
            extra = memory.reserve(memory.max_bytes - memory.used[0])
        try:
            with pytest.raises((ValueError, OSError, TimeoutError)):
                grants.lend(delivery, refs)
        finally:
            if extra:
                extra.close()
        assert not list(grants.root.iterdir()) and memory.used[0] == before
        delivery.close()  # Every failed borrow has been returned.


def test_failed_unlink_keeps_storage_and_metadata_until_retry(
    tmp_path, delivered, monkeypatch
):
    delivery, ref, disk, memory, _ = delivered
    grants = FileGrants(root=tmp_path, memory=memory)
    grant = grants.lend(delivery, [ref])
    before = memory.used[0]
    real_unlink = Path.unlink

    def denied(path, *args, **kwargs):
        if path.parent == grants.root:
            raise OSError("unlink failed")
        return real_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", denied)
        with pytest.raises(OSError, match="unlink failed"):
            grant.close()
        assert memory.used[0] == before and disk.used[0] == ref.size_bytes
        with pytest.raises(RuntimeError, match="consumers"):
            delivery.close()
        with pytest.raises(RuntimeError, match="consumers"):
            grants.close()
    grant.close()
    grants.close()


def test_failed_rollback_is_reachable_for_owner_cleanup(
    tmp_path, delivered, monkeypatch
):
    delivery, ref, _, memory, _ = delivered
    grants = FileGrants(root=tmp_path, memory=memory)
    real_unlink = Path.unlink

    def denied(path, *args, **kwargs):
        if path.parent == grants.root:
            raise OSError("rollback failed")
        return real_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", denied)
        with pytest.raises(OSError, match="rollback failed"):
            grants.lend(delivery, [ref, replace(ref, id="missing")])
        with pytest.raises(RuntimeError, match="consumers"):
            delivery.close()
    grants.close()  # Retries only unpublished failed grants; none has a consumer.
    delivery.close()


@pytest.mark.parametrize("replacement", ["symlink", "size"])
def test_worker_refuses_symlink_or_wrong_size(tmp_path, delivered, replacement):
    delivery, ref, _, memory, _ = delivered
    with (
        FileGrants(root=tmp_path, memory=memory) as grants,
        grants.lend(delivery, [ref]),
    ):
        path = resolve_file(grants.root, ref)
        path.unlink()
        if replacement == "symlink":
            path.symlink_to(tmp_path / "outside")
        else:
            path.write_bytes(b"x")
        with pytest.raises(ValueError, match="differs"):
            resolve_file(grants.root, ref)
