"""Public retained downloads over real interrupted HTTP streams, without workers."""

import hashlib
import io
import json
import os
import shlex
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import replace
from threading import Event

import pytest

from engine_tests.server.test_http import TOKEN, endpoint
from shadowspill_client import ServiceClient
from shadowspill_client.downloads import PROGRESS_BYTES, partial_path
from shadowspill_client.protocol.codec import ProtocolError
from shadowspill_client.protocol.requests import ArtifactRef
from shadowspill_client.transport.http import TransportError

PAYLOAD = bytes(range(256)) * 12289
REFERENCE = ArtifactRef("resumable", hashlib.sha256(PAYLOAD).hexdigest(), len(PAYLOAD))
CUTOFF = (2 << 20) + 117


def connect(ctx):
    return ServiceClient(
        f"http://127.0.0.1:{ctx.service.server_port}", token=TOKEN, max_connections=2
    )


def interrupted(ctx, monkeypatch, *, block=None, release=None):
    ctx.artifacts.put(REFERENCE, lambda _: io.BytesIO(PAYLOAD))
    original = ctx.artifacts.open
    observations = []

    @contextmanager
    def opened(reference):
        with original(reference) as stream:
            row = {"offset": None, "bytes": 0}
            observations.append(row)

            class Source:
                def seek(self, offset):
                    row["offset"] = offset
                    return stream.seek(offset)

                def read(self, size):
                    if block is not None:
                        block.set()
                        assert release.wait(5)
                    if len(observations) == 1 and block is None:
                        size = min(size, max(0, CUTOFF - stream.tell()))
                    result = stream.read(size)
                    row["bytes"] += len(result)
                    return result

            yield Source()

    monkeypatch.setattr(ctx.artifacts, "open", opened)
    return observations


def fail_copy(ctx, destination):
    with connect(ctx) as client:
        with pytest.raises((ProtocolError, TransportError)):
            client.artifact(REFERENCE).download(destination, resume=True).result(10)
        assert not destination.exists()
        directory = partial_path(destination)
        assert (directory / "data").stat().st_size == CUTOFF
        assert (directory / "progress.json").stat().st_size <= PROGRESS_BYTES
    assert client._memory.used == client._work.pending.used == (0, 0)


def test_resume_after_connection_and_process_exit_fetches_only_verified_suffix(
    tmp_path, monkeypatch
):
    with endpoint(tmp_path) as ctx:
        observations = interrupted(ctx, monkeypatch)
        destination = tmp_path / "copy.bin"
        fail_copy(ctx, destination)
        directory = partial_path(destination)
        # The lock file may survive abrupt process exit; ownership must not.
        child = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                """
import fcntl, os, sys
with open(sys.argv[1], 'r+b') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os._exit(7)
""",
                str(directory / "lock"),
            ],
            timeout=5,
        )
        assert child.returncode == 7
        # Simulate bytes written after the last progress commit and a stale temp.
        with (directory / "data").open("ab") as stream:
            stream.write(b"uncommitted")
        (directory / "progress.pending").write_bytes(b"interrupted metadata write")
        with connect(ctx) as client:
            assert (
                client.artifact(REFERENCE).download(destination, resume=True).result(10)
                == destination
            )
            assert destination.read_bytes() == PAYLOAD
            assert not directory.exists()
            assert observations == [
                {"offset": 0, "bytes": CUTOFF},
                {"offset": CUTOFF, "bytes": len(PAYLOAD) - CUTOFF},
            ]
            with pytest.raises(FileExistsError):
                client.artifact(REFERENCE).download(destination, resume=True).result(5)
            assert destination.read_bytes() == PAYLOAD and len(observations) == 2
        assert client._memory.used == client._work.pending.used == (0, 0)


@pytest.mark.parametrize(
    "fault",
    ["reference", "prefix", "short", "oversized", "metadata", "extra", "symlink"],
)
def test_corrupt_or_unrelated_partials_are_preserved_and_rejected_before_network(
    tmp_path, monkeypatch, fault
):
    with endpoint(tmp_path) as ctx:
        observations = interrupted(ctx, monkeypatch)
        destination = tmp_path / "copy.bin"
        fail_copy(ctx, destination)
        directory, reference = partial_path(destination), REFERENCE
        data = directory / "data"
        if fault == "reference":
            reference = replace(reference, id="different-artifact")
        elif fault == "prefix":
            with data.open("r+b") as stream:
                stream.write(b"corrupted")
        elif fault == "short":
            with data.open("r+b") as stream:
                stream.truncate(17)
        elif fault == "oversized":
            with data.open("r+b") as stream:
                stream.truncate(REFERENCE.size_bytes + 1)
        elif fault == "metadata":
            (directory / "progress.json").write_bytes(b"x" * (PROGRESS_BYTES + 1))
        elif fault == "extra":
            (directory / "unrelated").write_bytes(b"preserve me")
        else:
            unrelated = tmp_path / "unrelated.bin"
            unrelated.write_bytes(b"preserve me")
            data.unlink()
            data.symlink_to(unrelated)
        before = data.read_bytes()
        with connect(ctx) as client, pytest.raises((ValueError, OSError)):
            client.artifact(reference).download(destination, resume=True).result(5)
        assert len(observations) == 1 and not destination.exists()
        assert data.read_bytes() == before and directory.exists()
        assert client._memory.used == client._work.pending.used == (0, 0)


def test_concurrent_public_download_cannot_mutate_another_active_partial(
    tmp_path, monkeypatch
):
    entered, release = Event(), Event()
    with endpoint(tmp_path) as ctx:
        observations = interrupted(ctx, monkeypatch, block=entered, release=release)
        destination = tmp_path / "copy.bin"
        with connect(ctx) as first, connect(ctx) as second:
            active = first.artifact(REFERENCE).download(destination, resume=True)
            try:
                assert entered.wait(5)
                with pytest.raises(RuntimeError, match="active writer"):
                    second.artifact(REFERENCE).download(
                        destination, resume=True
                    ).result(5)
            finally:
                release.set()
            assert (
                active.result(10) == destination and destination.read_bytes() == PAYLOAD
            )
        assert len(observations) == 1 and not partial_path(destination).exists()


def test_completed_verified_partial_can_promote_without_another_server_read(
    tmp_path, monkeypatch
):
    from shadowspill_client import downloads

    with endpoint(tmp_path) as ctx:
        ctx.artifacts.put(REFERENCE, lambda _: io.BytesIO(PAYLOAD))
        destination = tmp_path / "copy.bin"
        original = os.link

        def unavailable(*args, **kwargs):
            raise OSError("injected destination publication failure")

        with connect(ctx) as client:
            with monkeypatch.context() as patch:
                patch.setattr(downloads.os, "link", unavailable)
                with pytest.raises(OSError, match="publication failure"):
                    client.artifact(REFERENCE).download(
                        destination, resume=True
                    ).result(10)
            assert not destination.exists() and os.link is original
            before = len(ctx.entered)
            assert (
                client.artifact(REFERENCE).download(destination, resume=True).result(10)
                == destination
            )
            assert len(ctx.entered) == before and destination.read_bytes() == PAYLOAD
            assert not partial_path(destination).exists()


@pytest.mark.parametrize("corrupt_completed", [False, True])
def test_checkpoint_resume_keeps_completed_rank_and_publishes_manifest_last(
    tmp_path, monkeypatch, corrupt_completed
):
    from client_tests.test_checkpoints import manifest as sample_manifest
    from shadowspill_client.protocol.checkpoints import CHECKPOINT_MEDIA_TYPE
    from shadowspill_client.protocol.codec import decode, encode

    with endpoint(tmp_path) as ctx:
        sample = sample_manifest()
        payloads = [PAYLOAD[:750001], PAYLOAD[:1500003]]
        rows = []
        for row, payload in zip(sample.ranks, payloads, strict=True):
            archive = replace(
                row.archive,
                size_bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
            )
            ctx.artifacts.put(archive, lambda _, payload=payload: io.BytesIO(payload))
            rows.append(replace(row, archive=archive))
        manifest = replace(sample, ranks=rows)
        body = encode(manifest)
        reference = ArtifactRef(
            "checkpoint",
            hashlib.sha256(body).hexdigest(),
            len(body),
            CHECKPOINT_MEDIA_TYPE,
        )
        ctx.artifacts.put(reference, lambda _: io.BytesIO(body))
        original = ctx.artifacts.open
        requests = []

        @contextmanager
        def interrupted_rank(ref):
            with original(ref) as source:
                requests.append(ref.id)
                if ref == rows[1].archive and requests.count(ref.id) == 1:
                    yield io.BytesIO(payloads[1][:131189])
                else:
                    yield source

        monkeypatch.setattr(ctx.artifacts, "open", interrupted_rank)
        destination = tmp_path / "checkpoint-copy"
        with connect(ctx) as client, pytest.raises((ProtocolError, TransportError)):
            client.artifact(reference).download(destination, resume=True).result(10)
        assert not (destination / "manifest.json").exists()
        complete = destination / "ranks/rank-00000.ssp"
        assert complete.read_bytes() == payloads[0]
        assert (
            partial_path(destination / "ranks/rank-00001.ssp") / "data"
        ).stat().st_size == 131189
        if corrupt_completed:
            with complete.open("r+b") as stream:
                stream.write(b"corrupt")
        with connect(ctx) as client:
            if corrupt_completed:
                with pytest.raises(ProtocolError, match="prefix differs"):
                    client.artifact(reference).download(
                        destination, resume=True
                    ).result(10)
                assert not (destination / "manifest.json").exists()
                assert requests == [
                    reference.id,
                    rows[0].archive.id,
                    rows[1].archive.id,
                    reference.id,
                ]
            else:
                assert (
                    client.artifact(reference)
                    .download(destination, resume=True)
                    .result(10)
                    == destination
                )
                assert decode((destination / "manifest.json").read_bytes()) == manifest
                assert {path.name for path in destination.iterdir()} == {
                    "manifest.json",
                    "ranks",
                }
                for row, payload in zip(rows, payloads, strict=True):
                    assert (
                        destination / "ranks" / f"rank-{row.rank:05d}.ssp"
                    ).read_bytes() == payload
                assert requests == [
                    reference.id,
                    rows[0].archive.id,
                    rows[1].archive.id,
                    reference.id,
                    rows[1].archive.id,
                ]
                with pytest.raises(FileExistsError):
                    client.artifact(reference).download(
                        destination, resume=True
                    ).result(5)
        assert client._memory.used == client._work.pending.used == (0, 0)


def test_empty_initial_partial_recovers_without_trusting_unknown_payload(tmp_path):
    with endpoint(tmp_path) as ctx:
        ctx.artifacts.put(REFERENCE, lambda _: io.BytesIO(PAYLOAD))
        destination = tmp_path / "copy.bin"
        directory = partial_path(destination)
        directory.mkdir()
        (directory / "data").touch()
        with connect(ctx) as client:
            assert (
                client.artifact(REFERENCE).download(destination, resume=True).result(10)
                == destination
            )
            assert destination.read_bytes() == PAYLOAD and not directory.exists()


@pytest.mark.skipif(
    not os.environ.get("SHADOWSPILL_DEMO_CLIENT_DOWNLOAD_SCRIPT"),
    reason="requires the prepared separate NumPy-only demo client",
)
def test_separate_demo_client_resumes_over_a_new_ssh_connection(tmp_path, monkeypatch):
    with endpoint(tmp_path) as ctx:
        observations = interrupted(ctx, monkeypatch)
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
                        os.environ["SHADOWSPILL_DEMO_CLIENT_DOWNLOAD_SCRIPT"],
                    ]
                ),
            ],
            input=json.dumps(
                {
                    "ssh_host": "chicago",
                    "port": ctx.service.server_port,
                    "token": TOKEN,
                    "reference": REFERENCE.to_dict(),
                    "cutoff": CUTOFF,
                }
            ),
            text=True,
            capture_output=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        receipt = json.loads(result.stdout)
        assert receipt["client_host"] == "tubingen" and all(receipt["absent"].values())
        assert receipt["both_connections_released"]
        assert observations == [
            {"offset": 0, "bytes": CUTOFF},
            {"offset": CUTOFF, "bytes": len(PAYLOAD) - CUTOFF},
        ]
        (tmp_path / "demo-download.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
