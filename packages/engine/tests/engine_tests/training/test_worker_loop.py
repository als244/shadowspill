"""Execution tickets, private stream faults and real child-process loss."""

import io
import select
import struct
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine_tests.server.test_request_records import request
from shadowspill.training.worker.channel import read_message, write_message
from shadowspill.training.worker.loop import WorkerLoop
from shadowspill.training.worker.protocol import (
    WorkerCall,
    WorkerControl,
    WorkerOutputError,
    WorkerReply,
    WorkerScope,
    ticket,
)
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.codec import ProtocolError
from shadowspill_client.protocol.requests import OperationRequest

SCOPE = WorkerScope("run", "server", "process-1", 0)


def call(invocation=1, sequence=1):
    return WorkerCall(
        SCOPE,
        invocation,
        request(f"request-{sequence}", sequence),
        "candidate",
        False,
        False,
    )


def loop(*, stage=None, execute=None):
    events = []

    def bind(command):
        def verify(budget):
            events.append(("stage", command.invocation))
            if stage:
                stage(command)

        return SimpleNamespace(
            command=command, checkpoint=None, restore=None, verify_inputs=verify
        )

    def run(bound):
        events.append(("execute", bound.command.invocation))
        if execute:
            execute(bound.command)

    return WorkerLoop(SCOPE, bind, run, BufferBudget(1)), events


def test_execution_requires_ready_ticket_and_separate_contiguous_invocations():
    worker, events = loop()
    first = call(sequence=7)  # Rejected submissions need not reach this worker.
    ready = worker.handle(first)
    assert ready == WorkerReply(ticket(first), "ready", "not_started")
    assert events == [("stage", 1)]
    completed = worker.handle(WorkerControl(ready.ticket, "execute"))
    assert completed == WorkerReply(ticket(first), "completed", "completed")
    second = call(2, 11)
    ready = worker.handle(second)
    assert worker.handle(WorkerControl(ready.ticket, "abort")).status == "aborted"
    third = call(3, 12)
    ready = worker.handle(third)
    assert worker.handle(WorkerControl(ready.ticket, "execute")).status == "completed"
    assert events == [
        ("stage", 1),
        ("execute", 1),
        ("stage", 2),
        ("stage", 3),
        ("execute", 3),
    ]
    assert worker.valid


def test_close_returns_one_final_reply_and_retires_the_private_channel():
    worker, events = loop()
    command = call()
    command = replace(
        command,
        candidate=None,
        request=OperationRequest(command.request.identity, "close"),
    )
    incoming, outgoing = io.BytesIO(), io.BytesIO()
    write_message(incoming, command)
    write_message(incoming, WorkerControl(ticket(command), "execute"))
    write_message(incoming, call(2, 2))  # Must remain unread after close.
    incoming.seek(0)
    worker.serve(incoming, outgoing)
    outgoing.seek(0)
    assert read_message(outgoing).status == "ready"
    assert read_message(outgoing).status == "completed"
    assert read_message(outgoing) is None and not worker.valid
    assert events == [("stage", 1), ("execute", 1)]


@pytest.mark.parametrize("point", ["before_stage", "completed"])
def test_duplicate_or_premature_execute_never_runs_again(point):
    worker, events = loop()
    command = call()
    if point != "before_stage":
        worker.handle(command)
    control = WorkerControl(ticket(command), "execute")
    if point == "completed":
        worker.handle(control)
    with pytest.raises(ProtocolError, match="does not match"):
        worker.handle(control)
    assert sum(event == "execute" for event, _ in events) == (point != "before_stage")
    assert not worker.valid
    with pytest.raises(RuntimeError, match="never replay"):
        worker.handle(call(2, 2))


@pytest.mark.parametrize(
    "changed",
    [
        lambda value: replace(value, invocation=2),
        lambda value: replace(value, scope=replace(SCOPE, worker_incarnation="old")),
    ],
)
def test_scope_and_invocation_gaps_invalidate_without_execution(changed):
    worker, events = loop()
    with pytest.raises(ProtocolError, match="stale, repeated"):
        worker.handle(changed(call()))
    assert not events and not worker.valid


def test_a_second_ready_call_or_replayed_submission_sequence_is_fenced():
    worker, events = loop()
    ready = worker.handle(call())
    with pytest.raises(ProtocolError, match="stale, repeated"):
        worker.handle(call())
    assert events == [("stage", 1)] and not worker.valid
    worker, events = loop()
    worker.handle(call())
    worker.handle(WorkerControl(ready.ticket, "execute"))
    with pytest.raises(ProtocolError, match="stale, repeated"):
        worker.handle(call(2, 1))
    assert events == [("stage", 1), ("execute", 1)]


def test_staging_failure_is_known_not_started_and_corrected_work_can_continue():
    def fail_first(command):
        if command.invocation == 1:
            raise OSError("bad local input " * 1000)

    worker, events = loop(stage=fail_first)
    reply = worker.handle(call())
    assert reply.status == "failed" and reply.effect == "not_started"
    assert len(reply.message) <= 1024 and worker.valid
    ready = worker.handle(call(2, 2))
    worker.handle(WorkerControl(ready.ticket, "execute"))
    assert events == [("stage", 1), ("stage", 2), ("execute", 2)]


def test_unclassified_execute_exception_has_unknown_effects_and_stops_live_worker():
    def fail(command):
        raise OSError("device lost")

    worker, events = loop(execute=fail)
    ready = worker.handle(call())
    reply = worker.handle(WorkerControl(ready.ticket, "execute"))
    assert reply.status == "failed" and reply.effect == "unknown"
    assert not worker.valid and events == [("stage", 1), ("execute", 1)]
    with pytest.raises(RuntimeError, match="never replay"):
        worker.handle(call(2, 2))


def test_confirmed_publication_failure_keeps_state_and_consumes_the_invocation():
    def publish(command):
        if command.invocation == 1:
            raise WorkerOutputError("output disk full")

    worker, events = loop(execute=publish)
    ready = worker.handle(call())
    reply = worker.handle(WorkerControl(ready.ticket, "execute"))
    assert reply.status == "failed" and reply.effect == "completed"
    assert reply.message == "output disk full" and worker.valid
    ready = worker.handle(call(2, 2))
    assert worker.handle(WorkerControl(ready.ticket, "execute")).status == "completed"
    assert events == [("stage", 1), ("execute", 1), ("stage", 2), ("execute", 2)]


def test_control_digest_cannot_release_different_ready_inputs():
    worker, events = loop()
    ready = worker.handle(call())
    changed = replace(ready.ticket, call_sha256="0" * 64)
    with pytest.raises(ProtocolError, match="does not match"):
        worker.handle(WorkerControl(changed, "execute"))
    assert events == [("stage", 1)]


class Fragmented(io.BytesIO):
    def read(self, size=-1):
        return super().read(min(size, 3))

    def write(self, data):
        return super().write(data[:2])


def test_private_channel_preserves_records_across_partial_reads_and_writes():
    stream = Fragmented()
    messages = [call(), WorkerControl(ticket(call()), "abort")]
    for message in messages:
        write_message(stream, message)
    stream.seek(0)
    assert [read_message(stream), read_message(stream), read_message(stream)] == [
        *messages,
        None,
    ]


def test_control_prefix_is_checked_before_body_reads():
    class Reader:
        def read(self, size):
            assert size == 4
            return struct.pack("!I", 1 << 30)

    with pytest.raises(ProtocolError, match="record bound"):
        read_message(Reader())
    for data in (b"\0", struct.pack("!I", 30) + b"short", struct.pack("!I", 0)):
        with pytest.raises(ProtocolError):
            read_message(io.BytesIO(data))


@pytest.mark.parametrize("written", [None, 0, -1, 1000000000])
def test_invalid_write_progress_cannot_spin_or_claim_delivery(written):
    class Writer:
        def write(self, data):
            return written

    with pytest.raises(ProtocolError, match="write progress"):
        write_message(Writer(), call())


def test_serve_never_keeps_mutable_worker_reusable_after_eof_or_broken_reply():
    incoming = io.BytesIO()
    write_message(incoming, call())
    incoming.seek(0)
    worker, events = loop()
    worker.serve(incoming, io.BytesIO())
    assert not worker.valid and events == [("stage", 1)]

    class Broken(io.BytesIO):
        def write(self, data):
            raise BrokenPipeError("lost receiver")

    incoming.seek(0)
    worker, events = loop()
    with pytest.raises(BrokenPipeError):
        worker.serve(incoming, Broken())
    assert not worker.valid


@pytest.mark.parametrize(
    "mode", ["normal", "lost_completion", "duplicate_execute", "close"]
)
def test_actual_child_process_channel_and_lost_completion_never_replay(tmp_path, mode):
    marker = tmp_path / "executions"
    process = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).with_name("_worker_loop.py")),
            str(marker),
            mode,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    try:
        command = call()
        if mode == "close":
            command = replace(
                command,
                candidate=None,
                request=OperationRequest(command.request.identity, "close"),
            )
        write_message(process.stdin, command)
        assert select.select([process.stdout], [], [], 10)[0]
        ready = read_message(process.stdout)
        assert ready.status == "ready" and not marker.exists()
        control = WorkerControl(ready.ticket, "execute")
        write_message(process.stdin, control)
        assert select.select([process.stdout], [], [], 10)[0]
        completed = read_message(process.stdout)
        assert marker.read_text() == "1\n"
        if mode == "lost_completion":
            assert completed is None
            assert process.wait(timeout=10) == 42
        else:
            assert completed.status == "completed"
            if mode == "duplicate_execute":
                write_message(process.stdin, control)
                assert process.wait(timeout=10) != 0
                assert read_message(process.stdout) is None
            else:
                process.stdin.close()
                assert process.wait(timeout=10) == 0, process.stderr.read().decode()
        assert marker.read_text() == "1\n"
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)
        for stream in (process.stdin, process.stdout, process.stderr):
            stream.close()
