"""Installed encoders publish bounded files without another whole-file copy."""

import hashlib
import io
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event

import numpy as np
import pytest

from engine_tests.server.test_artifacts import reference, store
from shadowspill.orchestration.launch.publication import OutputSpec
from shadowspill.orchestration.launch.staging import FileDelivery
from shadowspill.server.artifact_writer import ProducedArtifacts
from shadowspill.server.artifacts import INDEX_BYTES
from shadowspill.training.inputs.frames import InputFrame
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.payload import MEDIA_TYPE
from shadowspill_client.protocol.payload_io import read_payload


def spec(ref):
    return OutputSpec(ref.id, ref.size_bytes, ref.media_type)


def encode(writer):
    writer.write(memoryview(b"result"))
    return reference()


def test_existing_binary_encoder_publishes_directly_without_file_delivery_copy(
    tmp_path, monkeypatch
):
    values = np.arange(100000, dtype=np.float32)
    path = ("inputs", "values")
    frame = InputFrame({path: values.shape}, {path: "float32"})
    memory = BufferBudget(8 << 20, 10)

    def no_copy(*args):
        pytest.fail("installed result encoder must not copy a second complete file")

    monkeypatch.setattr(FileDelivery, "receive", no_copy)
    with store(tmp_path, memory=memory) as artifacts:
        ref = artifacts.produce(
            OutputSpec("encoded", frame.size.wire, MEDIA_TYPE),
            lambda stream: frame.write(stream, {path: values}, memory, "encoded"),
        )
        with (
            artifacts.open(ref) as stream,
            read_payload(stream, ref, memory, limits=frame.size) as decoded,
        ):
            np.testing.assert_array_equal(
                decoded.datums[0].inputs["values"].array(), values
            )
        assert artifacts._disk.used == (0, 0) and memory.used == (INDEX_BYTES, 1)
        assert not list(artifacts._scratch.iterdir())
    assert memory.used == (0, 0)
    with store(tmp_path) as reopened:
        assert reopened.get("encoded") == ref
        with reopened.open(ref) as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == ref.sha256


@pytest.mark.parametrize("issue", ["short", "extra", "digest", "identity", "error"])
def test_failed_encoder_never_publishes_partial_or_misidentified_bytes(tmp_path, issue):
    memory = BufferBudget(8 << 20, 10)

    def write(writer):
        if issue == "error":
            raise OSError("encoder failed")
        writer.write(
            memoryview({"short": b"resul", "extra": b"result!"}.get(issue, b"result"))
        )
        return replace(
            reference(),
            **(
                {"sha256": "0" * 64}
                if issue == "digest"
                else {"id": "other"}
                if issue == "identity"
                else {}
            ),
        )

    with store(tmp_path, memory=memory) as artifacts:
        with pytest.raises((ValueError, OSError)):
            artifacts.produce(spec(reference()), write)
        assert artifacts.get("output") is None and artifacts.used == (0, 0)
        assert artifacts._disk.used == (0, 0) and memory.used == (INDEX_BYTES, 1)
        assert artifacts.produce(spec(reference()), encode) == reference()


def test_existing_id_and_exhausted_capacity_do_not_invoke_an_encoder(tmp_path):
    def never(_):
        pytest.fail("unadmitted encoder")

    with store(tmp_path, max_bytes=6) as artifacts:
        artifacts.produce(spec(reference()), encode)
        with pytest.raises(ValueError, match="already committed"):
            artifacts.produce(spec(reference()), never)
        with pytest.raises(ValueError, match="capacity"):
            artifacts.produce(OutputSpec("another", 1, "other"), never)


def test_active_producer_is_not_visible_and_cannot_lose_its_reservations(tmp_path):
    entered, release = Event(), Event()
    memory = BufferBudget(8 << 20, 10)

    def blocked(writer):
        entered.set()
        assert release.wait(5)
        return encode(writer)

    with store(tmp_path, memory=memory) as artifacts, ThreadPoolExecutor(1) as pool:
        future = pool.submit(artifacts.produce, spec(reference()), blocked)
        try:
            assert entered.wait(5)
            assert artifacts.get("output") is None and artifacts.used == (6, 1)
            assert memory.used[0] > INDEX_BYTES and artifacts._disk.used == (6, 1)
            with pytest.raises(RuntimeError, match="producers"):
                artifacts.close()
        finally:
            release.set()
        assert future.result(timeout=5) == reference()
    assert memory.used == (0, 0)


def test_producer_cleanup_failure_retains_committed_bytes_and_quota_owner(
    tmp_path, monkeypatch
):
    memory = BufferBudget(8 << 20, 10)
    artifacts = store(tmp_path, memory=memory)
    close = ProducedArtifacts.close

    def fail(_):
        raise OSError("cannot remove temporary producer")

    monkeypatch.setattr(ProducedArtifacts, "close", fail)
    with pytest.raises(OSError):
        artifacts.produce(spec(reference()), encode)
    assert artifacts._pending and artifacts._disk.used == (6, 1)
    with pytest.raises(OSError):
        artifacts.close()
    monkeypatch.setattr(ProducedArtifacts, "close", close)
    artifacts.close()
    assert memory.used == (0, 0)
    with store(tmp_path) as recovered:
        assert recovered.get("output") == reference()


def test_uncertain_producer_commit_recovers_without_invoking_the_encoder(
    tmp_path, monkeypatch
):
    artifacts = store(tmp_path)
    append = artifacts._journal.append

    def failed_reply(rows):
        append(rows)
        raise OSError("commit reply lost")

    monkeypatch.setattr(artifacts._journal, "append", failed_reply)
    with pytest.raises(OSError):
        artifacts.produce(spec(reference()), encode)
    artifacts.close()
    with store(tmp_path) as recovered:
        assert recovered.get("output") == reference()
        with recovered.open(reference()) as stream:
            assert stream.read() == b"result"


def test_recovery_removes_only_an_unpublished_production_directory(tmp_path):
    with store(tmp_path) as artifacts:
        artifacts.put(reference(), lambda _: io.BytesIO(b"result"))
        scratch = artifacts._scratch
    abandoned = scratch / "production-abandoned"
    abandoned.mkdir()
    (abandoned / "payload").write_bytes(b"partial")
    with store(tmp_path) as recovered:
        assert not abandoned.exists()
        assert recovered.get("output") == reference()
