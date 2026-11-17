"""Public artifact uploads over actual authenticated HTTP and private storage."""

import hashlib
import io
import json
import os
import shlex
import subprocess
from dataclasses import replace

import pytest

from client_tests.test_checkpoints import manifest as sample_manifest
from engine_tests.server.test_http import AUTH, OTHER_TOKEN, TOKEN, endpoint, exchange
from shadowspill.server.http import _Handler
from shadowspill_client import ServiceClient
from shadowspill_client.checkpoints import CheckpointHandle
from shadowspill_client.protocol.checkpoints import CHECKPOINT_MEDIA_TYPE
from shadowspill_client.protocol.codec import MAX_METADATA_BYTES, encode
from shadowspill_client.protocol.requests import ArtifactRef
from shadowspill_client.protocol.service import CONTROL_MEDIA_TYPE
from shadowspill_client.protocol.uploads import (
    MAX_UPLOAD_CHUNK_BYTES,
    ArtifactUpload,
    UploadStatus,
)
from shadowspill_client.transport.http import RemoteError, TransportError

PAYLOAD = bytes(range(256)) * 32769
REFERENCE = ArtifactRef(
    "upload." + hashlib.sha256(PAYLOAD).hexdigest(),
    hashlib.sha256(PAYLOAD).hexdigest(),
    len(PAYLOAD),
)


def connect(ctx, **kwargs):
    return ServiceClient(
        f"http://127.0.0.1:{ctx.service.server_port}",
        token=TOKEN,
        max_connections=2,
        **kwargs,
    )


def interrupted_chunks(ctx, monkeypatch):
    original = ctx.artifacts.uploads.chunk
    calls = []

    def receive(name, offset, size, sha256, source):
        calls.append((name, offset, size))
        if len(calls) == 2:
            incoming = source

            class Dropped:
                remaining = 131189

                def readinto(self, view):
                    if not self.remaining:
                        raise OSError("injected upload disconnect")
                    with view[: self.remaining] as part:
                        count = incoming.readinto(part)
                    self.remaining -= count
                    return count

            source = Dropped()
        return original(name, offset, size, sha256, source)

    monkeypatch.setattr(ctx.artifacts.uploads, "chunk", receive)
    return calls


def test_public_upload_reconnect_sends_only_after_verified_committed_prefix(
    tmp_path, monkeypatch
):
    path = tmp_path / "input.bin"
    path.write_bytes(PAYLOAD)
    with endpoint(tmp_path, artifact_bytes=16 << 20) as ctx:
        calls = interrupted_chunks(ctx, monkeypatch)
        with connect(ctx) as service, pytest.raises(TransportError):
            service.artifacts.upload(path).result(10)
        assert service._memory.used == service._work.pending.used == (0, 0)
        status = ctx.artifacts.uploads.get(REFERENCE.id)
        assert status.offset == MAX_UPLOAD_CHUNK_BYTES
        assert (
            ctx.artifacts.uploads._path(REFERENCE.id).stat().st_size
            == MAX_UPLOAD_CHUNK_BYTES + 131189
        )
        assert ctx.artifacts.get(REFERENCE.id) is None
        assert len(calls) == 2  # Failed future never implicitly restarts a transfer.
        with connect(ctx) as service:
            handle = service.artifacts.upload(path).result(10)
            assert handle.reference == REFERENCE
            destination = handle.download(tmp_path / "copy.bin").result(10)
            assert destination.read_bytes() == PAYLOAD
            assert service.artifacts.upload(path).result(10) == handle
        assert [row[1] for row in calls] == [
            0,
            MAX_UPLOAD_CHUNK_BYTES,
            MAX_UPLOAD_CHUNK_BYTES,
            2 * MAX_UPLOAD_CHUNK_BYTES,
        ]
        assert ctx.artifacts.used == (len(PAYLOAD), 1)
        assert service._memory.used == service._work.pending.used == (0, 0)


@pytest.mark.parametrize("when", ["chunk", "complete"])
def test_lost_upload_acknowledgement_is_inspected_on_explicit_retry(
    tmp_path, monkeypatch, when
):
    path = tmp_path / "input.bin"
    path.write_bytes(PAYLOAD)
    lost = []
    original = _Handler._reply

    def reply(self, value, *args):
        if (
            isinstance(value, UploadStatus)
            and value.offset
            and not lost
            and ((when == "complete") == (value.state == "complete"))
        ):
            lost.append(value)
            raise OSError("lost HTTP acknowledgement")
        return original(self, value, *args)

    monkeypatch.setattr(_Handler, "_reply", reply)
    with endpoint(tmp_path, artifact_bytes=16 << 20) as ctx:
        with connect(ctx) as service, pytest.raises(TransportError):
            service.artifacts.upload(path).result(10)
        assert len(lost) == 1
        received = []
        chunk = ctx.artifacts.uploads.chunk

        def observe(*args):
            received.append(args[1])
            return chunk(*args)

        monkeypatch.setattr(ctx.artifacts.uploads, "chunk", observe)
        with connect(ctx) as service:
            assert service.artifacts.upload(path).result(10).reference == REFERENCE
        assert received == (
            []
            if when == "complete"
            else [MAX_UPLOAD_CHUNK_BYTES, 2 * MAX_UPLOAD_CHUNK_BYTES]
        )


def test_authentication_owner_isolation_offsets_and_abort_routes(tmp_path):
    body = b"result"
    ref = ArtifactRef("upload", hashlib.sha256(body).hexdigest(), len(body))
    request = ArtifactUpload(ref)
    with endpoint(tmp_path) as ctx:
        assert exchange(ctx, "POST", "/v1/uploads", body=encode(request))[0] == 403
        for value in (
            b"not json",
            encode(
                UploadStatus(ref, 0, hashlib.sha256().hexdigest(), "uploading", 100)
            ),
        ):
            assert (
                exchange(
                    ctx,
                    "POST",
                    "/v1/uploads",
                    headers=(AUTH, ("Content-Type", CONTROL_MEDIA_TYPE)),
                    body=value,
                )[0]
                == 400
            )
        status = ctx.client.request("POST", "/v1/uploads", request).value
        assert status.offset == 0
        with pytest.raises(RemoteError) as error:
            ctx.connect(OTHER_TOKEN).request("GET", "/v1/uploads/upload")
        assert error.value.status == 404
        for route in (
            "/v1/uploads/upload/chunks/01/" + ref.sha256,
            "/v1/uploads/upload/chunks/1/" + ref.sha256,
            "/v1/uploads/upload/chunks/0/bad",
        ):
            with pytest.raises(RemoteError) as error:
                ctx.client.upload(route, ref, (body,))
            assert error.value.status == 400
        assert ctx.client.request("GET", "/v1/uploads/upload").value == status
        assert ctx.client.request("DELETE", "/v1/uploads/upload").value == request
        assert ctx.artifacts.used == (0, 0)
    with endpoint(tmp_path, wrong_scope=True) as ctx:
        with pytest.raises(RemoteError) as error:
            ctx.connect(OTHER_TOKEN).request("POST", "/v1/uploads", request)
        assert error.value.status == 403 and ctx.artifacts.used == (0, 0)


def checkpoint_directory(path, payloads=(b"rank zero", b"rank one")):
    path.mkdir()
    (path / "ranks").mkdir()
    sample = sample_manifest()
    rows = []
    for row, data in zip(sample.ranks, payloads, strict=True):
        reference = replace(
            row.archive, size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest()
        )
        rows.append(replace(row, archive=reference))
        (path / "ranks" / f"rank-{row.rank:05d}.ssp").write_bytes(data)
    manifest = replace(sample, ranks=rows)
    body = encode(manifest)
    (path / "manifest.json").write_bytes(body)
    reference = ArtifactRef(
        "checkpoint", hashlib.sha256(body).hexdigest(), len(body), CHECKPOINT_MEDIA_TYPE
    )
    return manifest, reference


def test_checkpoint_directory_keeps_rank_ids_and_requires_every_published_member(
    tmp_path,
):
    path = tmp_path / "checkpoint"
    manifest, reference = checkpoint_directory(path)
    with endpoint(tmp_path) as ctx, connect(ctx) as service:
        # A manifest alone cannot become a checkpoint with missing ranks.
        with pytest.raises(RemoteError) as error:
            service.artifacts.upload(
                path / "manifest.json", reference=reference
            ).result(10)
        assert error.value.status == 400 and ctx.artifacts.get(reference.id) is None
        handle = service.artifacts.upload(path, reference=reference).result(10)
        assert isinstance(handle, CheckpointHandle) and handle.reference == reference
        assert handle.manifest().result(10) == manifest
        assert ctx.artifacts.page() == tuple(
            sorted(
                (reference, *(row.archive for row in manifest.ranks)),
                key=lambda ref: ref.id,
            )
        )
        target = handle.download(tmp_path / "copy").result(10)
        assert (target / "manifest.json").read_bytes() == (
            path / "manifest.json"
        ).read_bytes()
        for row in manifest.ranks:
            assert (target / "ranks" / f"rank-{row.rank:05d}.ssp").read_bytes() == (
                path / "ranks" / f"rank-{row.rank:05d}.ssp"
            ).read_bytes()
        # Configuration/Plans are not silently copied or required for a backup's
        # registration. Existing restore admission reports them before workers.
        from shadowspill.server.startup.checkpoint import (
            require_checkpoint_dependencies,
        )

        with pytest.raises(ValueError, match=manifest.info.configuration.id):
            require_checkpoint_dependencies(ctx.artifacts, manifest)
        before = ctx.artifacts.used
        assert service.artifacts.upload(path, reference=reference).result(10) == handle
        assert ctx.artifacts.used == before


@pytest.mark.parametrize(
    "damage",
    [
        "missing_rank",
        "short_rank",
        "bad_manifest",
        "huge_manifest",
        "symlink_rank",
        "symlink_directory",
    ],
)
def test_invalid_local_checkpoint_is_rejected_before_starting_any_upload(
    tmp_path, damage
):
    path = tmp_path / "checkpoint"
    checkpoint_directory(path)
    rank = path / "ranks/rank-00001.ssp"
    if damage in ("missing_rank", "symlink_rank"):
        rank.unlink()
        if damage == "symlink_rank":
            rank.symlink_to(path / "ranks/rank-00000.ssp")
    elif damage == "short_rank":
        rank.write_bytes(b"bad")
    elif damage == "bad_manifest":
        (path / "manifest.json").write_bytes(b"{}")
    elif damage == "huge_manifest":
        (path / "manifest.json").write_bytes(b"x" * (MAX_METADATA_BYTES + 1))
    else:
        moved = path / "moved-ranks"
        (path / "ranks").rename(moved)
        (path / "ranks").symlink_to(moved)
    with endpoint(tmp_path) as ctx, connect(ctx) as service:
        with pytest.raises((OSError, ValueError)):
            service.artifacts.upload(path).result(10)
        assert ctx.artifacts.used == (0, 0)


@pytest.mark.parametrize(
    "damage", ["checksum", "prefix", "during_transfer", "nonregular"]
)
def test_local_file_changes_or_bad_references_never_publish(
    tmp_path, monkeypatch, damage
):
    path = tmp_path / "input.bin"
    path.write_bytes(PAYLOAD)
    ref = REFERENCE
    with endpoint(tmp_path, artifact_bytes=16 << 20) as ctx, connect(ctx) as service:
        if damage == "checksum":
            ref = replace(ref, sha256="a" * 64)
        elif damage == "prefix":
            ref = replace(ref, sha256="a" * 64)
            ctx.artifacts.uploads.begin(ref)
            chunk = b"wrong"
            ctx.artifacts.uploads.chunk(
                ref.id,
                0,
                len(chunk),
                hashlib.sha256(chunk).hexdigest(),
                io.BytesIO(chunk),
            )
        elif damage == "nonregular":
            path.unlink()
            os.mkfifo(path)
        else:
            original = ctx.artifacts.uploads.chunk

            def changed(*args):
                result = original(*args)
                with path.open("r+b") as stream:
                    stream.write(b"changed")
                return result

            monkeypatch.setattr(ctx.artifacts.uploads, "chunk", changed)
        with pytest.raises(ValueError):
            service.artifacts.upload(path, reference=ref).result(10)
        assert ctx.artifacts.get(ref.id) is None


def test_empty_file_is_complete_without_a_binary_chunk(tmp_path, monkeypatch):
    path = tmp_path / "empty"
    path.touch()
    with endpoint(tmp_path) as ctx, connect(ctx) as service:
        monkeypatch.setattr(
            ctx.artifacts.uploads,
            "chunk",
            lambda *args: pytest.fail("empty file sent a chunk"),
        )
        handle = service.artifacts.upload(path).result(10)
        assert handle.reference.size_bytes == 0
        assert ctx.artifacts.get(handle.id) == handle.reference


def test_public_upload_resumes_after_the_service_and_store_reopen(
    tmp_path, monkeypatch
):
    path = tmp_path / "input.bin"
    path.write_bytes(PAYLOAD)
    with endpoint(tmp_path, artifact_bytes=16 << 20) as ctx:
        interrupted_chunks(ctx, monkeypatch)
        with connect(ctx) as service, pytest.raises(TransportError):
            service.artifacts.upload(path).result(10)
    with endpoint(tmp_path, artifact_bytes=16 << 20) as ctx:
        assert ctx.artifacts.uploads.get(REFERENCE.id).offset == MAX_UPLOAD_CHUNK_BYTES
        assert (
            ctx.artifacts.uploads._path(REFERENCE.id).stat().st_size
            == MAX_UPLOAD_CHUNK_BYTES
        )
        with connect(ctx) as service:
            assert service.artifacts.upload(path).result(10).reference == REFERENCE


def real_generation(artifacts):
    from engine_tests.server.test_artifacts import reference as artifact_reference
    from engine_tests.server.test_checkpoint_restore import configuration
    from engine_tests.server.test_checkpoints import boundary, stream_rank, workers
    from shadowspill.server.checkpoints import publish_checkpoint

    info = boundary(artifacts)
    config = configuration()
    raw = encode(config)
    config_ref = artifact_reference(raw, "resolved-configuration")
    artifacts.put(config_ref, lambda _: io.BytesIO(raw))
    info = replace(info, configuration=config_ref)
    executors, ranks = workers(info)
    reference = publish_checkpoint(
        artifacts,
        info,
        ranks,
        lambda rank, sink: stream_rank(executors[rank.rank], info, sink),
    )
    return config, info, executors, reference


def test_real_checkpoint_download_upload_to_fresh_storage_restores_every_cpu_tensor(
    tmp_path,
):
    from engine_tests.server.test_checkpoint_restore import compare_saved
    from shadowspill.server.startup.checkpoint import CheckpointResolver

    source_root, target_root = tmp_path / "source", tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    with endpoint(source_root) as source, connect(source) as client:
        config, info, executors, reference = real_generation(source.artifacts)
        local = client.artifact(reference).download(tmp_path / "backup").result(10)
        for ref in (info.configuration, info.plans):
            client.artifact(ref).download(tmp_path / ref.id).result(10)
    with endpoint(target_root) as target, connect(target) as client:
        handle = client.artifacts.upload(local, reference=reference).result(10)
        assert isinstance(handle, CheckpointHandle)
        resolver = CheckpointResolver(
            target.artifacts, session_id="other", world_size=2, configuration=config
        )
        for dependency in (info.configuration, info.plans):
            with pytest.raises(ValueError, match=dependency.id):
                resolver.resolve(handle.reference)
            copied = client.artifacts.upload(
                tmp_path / dependency.id, reference=dependency
            ).result(10)
            assert copied.reference == dependency
        resolved = resolver.resolve(handle.reference)
        restored = compare_saved(
            target.artifacts, handle.reference, executors, tmp_path
        )
        assert restored == resolved.manifest
        assert restored.info.completed_updates == 9
        assert restored.info.metadata["cursor"] == 17
        assert target.artifacts.used[1] == 5  # Two ranks, manifest, config and Plans.


@pytest.mark.skipif(
    not os.environ.get("SHADOWSPILL_DEMO_CLIENT_UPLOAD_SCRIPT"),
    reason="requires the prepared separate NumPy-only demo client",
)
def test_separate_demo_client_resumes_upload_and_copies_real_checkpoint(
    tmp_path, monkeypatch
):
    from engine_tests.server.test_checkpoint_restore import compare_saved
    from shadowspill.server.startup.checkpoint import CheckpointResolver

    source_root, target_root = tmp_path / "source", tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    with (
        endpoint(source_root) as source,
        endpoint(target_root, artifact_bytes=16 << 20) as target,
    ):
        config, info, executors, checkpoint = real_generation(source.artifacts)
        calls = interrupted_chunks(target, monkeypatch)
        result = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                os.environ["SHADOWSPILL_DEMO_CLIENT_HOST"],
                shlex.join(
                    [
                        os.environ["SHADOWSPILL_DEMO_CLIENT_PYTHON"],
                        "-I",
                        os.environ["SHADOWSPILL_DEMO_CLIENT_UPLOAD_SCRIPT"],
                    ]
                ),
            ],
            input=json.dumps(
                {
                    "ssh_host": "chicago",
                    "source_port": source.service.server_port,
                    "port": target.service.server_port,
                    "token": TOKEN,
                    "reference": REFERENCE.to_dict(),
                    "checkpoint": checkpoint.to_dict(),
                    "dependencies": [
                        info.configuration.to_dict(),
                        info.plans.to_dict(),
                    ],
                }
            ),
            text=True,
            capture_output=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        receipt = json.loads(result.stdout)
        assert receipt["client_host"] == "tubingen" and all(receipt["absent"].values())
        assert receipt["all_connections_released"]
        file_calls = [
            (offset, size) for name, offset, size in calls if name == REFERENCE.id
        ]
        assert [offset for offset, _ in file_calls] == [
            0,
            MAX_UPLOAD_CHUNK_BYTES,
            MAX_UPLOAD_CHUNK_BYTES,
            2 * MAX_UPLOAD_CHUNK_BYTES,
        ]
        resolved = CheckpointResolver(
            target.artifacts, session_id="copy", world_size=2, configuration=config
        ).resolve(checkpoint)
        assert (
            compare_saved(target.artifacts, checkpoint, executors, tmp_path)
            == resolved.manifest
        )
        assert target.artifacts.used[1] == 6
        assert all(owner == "owner" for owner in target.entered)
        receipt["file_chunks"] = file_calls
        receipt["all_cpu_tensors_restored"] = True
        (tmp_path / "demo-upload.json").write_text(json.dumps(receipt, indent=2) + "\n")
