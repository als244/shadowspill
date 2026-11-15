"""Bounded private bytes still require the sole worker loop's completion ticket."""

import hashlib
import io
import select
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from engine_tests.server.test_artifacts import store
from engine_tests.training.test_worker_loop import SCOPE, Fragmented, call
from shadowspill.orchestration.launch.publication import OutputSpec
from shadowspill.training.worker.binary import WorkerStream, receive_stream
from shadowspill.training.worker.channel import _write, read_message, write_message
from shadowspill.training.worker.connection import WorkerConnection
from shadowspill.training.worker.protocol import WorkerReply, ticket
from shadowspill_client.protocol.codec import ProtocolError

SIZE = (5 << 16) + 17


class FragmentedInput(Fragmented):
    def readinto(self, view):
        return super().readinto(view[:3])


@pytest.mark.parametrize("size", [0, 17, SIZE])
def test_fragmented_stream_retains_bounded_bytes_and_the_separate_control_reply(size):
    channel = FragmentedInput()
    payload = b"x" * size
    with WorkerStream(channel, size_bytes=size) as sender:
        sender.write(payload)
    reference = sender.reference("output", "application/octet-stream")
    reply = WorkerReply(ticket(call()), "completed", "completed", result=reference)
    write_message(channel, reply)
    channel.seek(0)
    destination = Fragmented()
    receive_stream(channel, destination, size_bytes=size)
    assert destination.getvalue() == payload
    assert reference.sha256 == hashlib.sha256(payload).hexdigest()
    assert read_message(channel) == reply
    assert read_message(channel) is None


@pytest.mark.parametrize("prefix", [65536, 0x80010001, 0x80000002])
def test_frame_kind_and_length_are_checked_before_allocating_or_reading_the_body(
    prefix,
):
    class Header:
        def read(self, size):
            assert size == 4
            return struct.pack("!I", prefix)

    with pytest.raises(ProtocolError, match="bound or kind"):
        receive_stream(Header(), io.BytesIO(), size_bytes=1)


def test_stream_and_control_frames_cannot_be_mistaken_for_each_other():
    channel = io.BytesIO()
    with WorkerStream(channel, size_bytes=1) as output:
        output.write(b"x")
    channel.seek(0)
    with pytest.raises(ProtocolError, match="control record bound"):
        read_message(channel)
    with pytest.raises(ProtocolError, match="declared size"):
        receive_stream(
            io.BytesIO(struct.pack("!I", 0x80000000)), io.BytesIO(), size_bytes=1
        )


def test_failed_encoder_preserves_its_diagnostic_and_cannot_claim_a_complete_stream():
    channel = io.BytesIO()
    with (
        pytest.raises(OSError, match="encoder failed"),
        WorkerStream(channel, size_bytes=3) as output,
    ):
        output.write(b"x")
        raise OSError("encoder failed")
    with pytest.raises(RuntimeError, match="not completed"):
        output.reference("output", "application/octet-stream")
    channel.seek(0)
    with pytest.raises(ProtocolError, match="declared size"):
        receive_stream(channel, io.BytesIO(), size_bytes=3)


@pytest.mark.parametrize("count", [None, False, 0, -1, 10])
def test_invalid_sink_progress_stops_transfer(count):
    class BadWriter:
        def write(self, part):
            return count

    channel = io.BytesIO()
    with WorkerStream(channel, size_bytes=1) as output:
        output.write(b"x")
    channel.seek(0)
    with pytest.raises(ProtocolError, match="write progress"):
        receive_stream(channel, BadWriter(), size_bytes=1)


@pytest.mark.parametrize("count", [None, False, 0, -1, 10])
def test_invalid_source_progress_stops_transfer(count):
    class InvalidInput(io.BytesIO):
        def readinto(self, view):
            return count

    channel = InvalidInput(struct.pack("!I", 0x80000001) + b"x")
    with pytest.raises(ProtocolError, match="read made no valid progress"):
        receive_stream(channel, io.BytesIO(), size_bytes=1)


def test_payload_uses_one_reusable_buffer_and_never_calls_read_for_tensor_bytes():
    channel = io.BytesIO()
    with WorkerStream(channel, size_bytes=SIZE) as output:
        output.write(b"x" * SIZE)
    allocations = set()

    class BoundedInput(io.BytesIO):
        def read(self, size):
            assert size == 4  # Control prefix only; no payload byte-string copy.
            return super().read(size)

        def readinto(self, view):
            assert len(view.obj) == 65536 and len(view) <= 65536
            allocations.add(id(view.obj))
            return super().readinto(view)

    receive_stream(BoundedInput(channel.getvalue()), io.BytesIO(), size_bytes=SIZE)
    assert len(allocations) == 1


def test_partial_final_write_cannot_claim_more_bytes_than_remain():
    class BadTail:
        def __init__(self):
            self.calls = 0

        def write(self, part):
            self.calls += 1
            return 65536 if self.calls == 1 else 2

    with pytest.raises(ProtocolError, match="write progress"):
        _write(BadTail(), b"x" * 65537)


@pytest.mark.parametrize(
    "mode", ["normal", "lost_stream", "lost_completion", "bad_reference"]
)
def test_real_worker_loop_streams_to_persistent_storage_and_never_replays(
    tmp_path, mode
):
    marker = tmp_path / "executions"
    process = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).with_name("_worker_stream.py")),
            str(marker),
            mode,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    connection = WorkerConnection(SCOPE, process.stdout, process.stdin)
    try:
        with store(tmp_path) as artifacts:
            for invocation in range(1, 3 if mode == "normal" else 2):
                command = call(invocation, invocation)
                connection.stage(command)
                assert select.select([process.stdout], [], [], 10)[0]
                assert connection.staged().status == "ready"
                before = "" if not marker.exists() else marker.read_text()
                assert before == ("" if invocation == 1 else "1\n")

                def produce(writer, invocation=invocation):
                    # Persistent bytes are admitted before releasing the worker.
                    connection.release("execute")
                    assert select.select([process.stdout], [], [], 10)[0]
                    connection.receive_stream(writer, size_bytes=SIZE)
                    assert artifacts.get(f"output-{invocation}") is None
                    assert select.select([process.stdout], [], [], 10)[0]
                    return connection.finished().result

                spec = OutputSpec(
                    f"output-{invocation}", SIZE, "application/octet-stream"
                )
                if mode == "normal":
                    saved = artifacts.produce(spec, produce)
                    assert (
                        connection.valid
                        and connection.next_invocation == invocation + 1
                    )
                    with artifacts.open(saved) as stream:
                        assert (
                            hashlib.file_digest(stream, "sha256").hexdigest()
                            == saved.sha256
                        )
                else:
                    with pytest.raises(ProtocolError):
                        artifacts.produce(spec, produce)
                    assert not connection.valid and artifacts.used == (0, 0)
                    with pytest.raises(RuntimeError, match="never resend"):
                        connection.stage(command)
            assert marker.read_text() == ("1\n2\n" if mode == "normal" else "1\n")
        if mode != "lost_stream" and mode != "lost_completion":
            connection.close()
            assert process.wait(timeout=10) == 0, process.stderr.read().decode()
        else:
            assert process.wait(timeout=10) == 42
    finally:
        connection.close()
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)
        process.stderr.close()
