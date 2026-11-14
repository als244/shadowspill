"""Actual private streams for all-worker release, abort and lost-process effects."""

import io
import socket
import subprocess
import sys
from contextlib import contextmanager, suppress
from dataclasses import replace
from pathlib import Path
from threading import Barrier, Thread
from types import SimpleNamespace

import pytest

from engine_tests.training.test_worker_commands import invocation as prepared_invocation
from engine_tests.training.test_worker_loop import SCOPE, call
from shadowspill.training.session.dispatch import ExecutionNotStarted
from shadowspill.training.worker.channel import write_message
from shadowspill.training.worker.connection import WorkerConnection
from shadowspill.training.worker.group import WorkerGroup
from shadowspill.training.worker.loop import WorkerLoop
from shadowspill.training.worker.protocol import WorkerOutputError, WorkerReply, ticket
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.codec import ProtocolError


def commands(invocation=1, sequence=1):
    return {
        rank: replace(call(invocation, sequence), scope=replace(SCOPE, rank=rank))
        for rank in range(2)
    }


@contextmanager
def workers(
    *,
    reject=None,
    execute=None,
    break_abort=False,
    count=2,
    binding=None,
    publication=None,
):
    connections, peers, threads, events, errors = {}, [], [], [], []
    barrier = Barrier(count)

    def start(rank):
        parent, child = socket.socketpair()
        parent.settimeout(5)
        child.settimeout(5)
        peers.extend((parent, child))
        parent_stream = parent.makefile("rwb", buffering=0)
        scope = replace(SCOPE, rank=rank)
        connections[rank] = WorkerConnection(scope, parent_stream, parent_stream)

        def bind(command):
            if binding:
                return binding(rank, command)

            def verify(budget):
                events.append((rank, "stage", command.invocation))
                if reject and reject(rank, command):
                    raise OSError("input delivery failed")

            return SimpleNamespace(command=command, verify_inputs=verify)

        def run(bound):
            events.append((rank, "execute", bound.command.invocation))
            if execute:
                execute(rank, bound.command)
            # A controller that waits for rank 0 before releasing rank 1 fails.
            barrier.wait(timeout=3)
            if publication:
                publication(rank, bound.command)

        worker = WorkerLoop(scope, bind, run, BufferBudget(16 << 20))
        original = worker.handle

        def handle(message):
            if break_abort and getattr(message, "action", None) == "abort":
                child.shutdown(socket.SHUT_RDWR)
                raise OSError("abort reply lost")
            return original(message)

        worker.handle = handle

        def serve():
            try:
                with child.makefile("rwb", buffering=0) as stream:
                    worker.serve(stream, stream)
            except (OSError, ValueError, RuntimeError) as error:
                errors.append(error)

        thread = Thread(target=serve)
        thread.start()
        threads.append(thread)

    for rank in range(count):
        start(rank)

    stopped = []

    def stop():
        stopped.append(True)
        barrier.abort()
        for peer in peers:
            with suppress(OSError):
                peer.shutdown(socket.SHUT_RDWR)
        for thread in threads:
            thread.join(timeout=5)
            assert not thread.is_alive()

    group = WorkerGroup(connections, stop=stop)
    try:
        yield group, events, stopped
    finally:
        group.close()
        for peer in peers:
            peer.close()


def test_every_worker_is_ready_and_released_before_collective_completion_waits():
    with workers() as (group, events, stopped):
        for invocation in (1, 2):
            result = group.run(commands(invocation, invocation * 3))
            assert all(reply.status == "completed" for reply in result.values())
            phases = [phase for _, phase, i in events if i == invocation]
            assert phases == ["stage", "stage", "execute", "execute"]
        assert not stopped
        assert all(item.next_invocation == 3 for item in group.connections.values())
    assert stopped == [True]


def test_all_ready_workers_abort_before_known_unstarted_failure_and_can_continue():
    with workers(
        reject=lambda rank, command: rank == 1 and command.invocation == 1
    ) as (group, events, stopped):
        with pytest.raises(ExecutionNotStarted, match="input delivery failed"):
            group.run(commands())
        assert all(phase == "stage" for _, phase, _ in events)
        assert group.valid and not stopped
        result = group.run(commands(2, 2))
        assert all(reply.status == "completed" for reply in result.values())
        assert sum(phase == "execute" for _, phase, _ in events) == 2


def test_publication_failure_waits_for_all_confirmed_effects_and_allows_new_work():
    def publish(rank, command):
        if rank == 0 and command.invocation == 1:
            raise WorkerOutputError("rank-zero output disk full")

    with workers(publication=publish) as (group, events, stopped):
        result = group.run(commands())
        assert result[0].status == "failed" and result[0].effect == "completed"
        assert result[1].status == "completed"
        assert group.valid and not stopped
        assert all(connection.valid for connection in group.connections.values())
        with pytest.raises(ExecutionNotStarted):
            group.run(commands())
        result = group.run(commands(2, 2))
        assert all(reply.status == "completed" for reply in result.values())
        assert sum(phase == "execute" for _, phase, _ in events) == 4


def test_dispatch_commits_worker_effects_before_reporting_worker_publication_failure(
    tmp_path,
):
    from engine_tests.server.test_request_records import store
    from engine_tests.training.test_request_dispatch import HostWork, publish, submit
    from shadowspill.training.session.dispatch import RequestDispatcher
    from shadowspill.training.session.state import OperationState

    def fail_output(rank, command):
        if rank == 0 and command.invocation == 1:
            raise WorkerOutputError("required worker output unavailable")

    with store(tmp_path) as records, workers(publication=fail_output) as (group, _, _):
        state = OperationState()
        dispatcher = RequestDispatcher(records, state, max_pending=2)
        submit(dispatcher, records, "a", 1)
        submit(dispatcher, records, "queued", 2)
        calls = []

        def execute(transition):
            calls.append(transition.request_id)
            request = records.get(transition.request_id)[0]
            work = HostWork()
            work.replies = group.run(
                {
                    rank: replace(command, request=request)
                    for rank, command in commands(
                        len(calls), request.identity.sequence
                    ).items()
                }
            )
            return work

        def output(receipt, work):
            assert records.execution(receipt.identity.request_id) == receipt
            for reply in work.replies.values():
                if reply.status == "failed":
                    raise OSError(reply.message)
            return publish(receipt, work)

        failed = dispatcher.run("a", execute, output, participates=True)
        assert failed.failure.code == "publication_failed"
        assert failed.failure.execution_effect == "completed"
        assert failed.failure.completed_updates == 1 and failed.failure.state_valid
        assert records.get("queued")[1].failure.failed_dependencies == ("a",)
        assert dispatcher.run("a", execute, output, participates=True) == failed
        assert calls == ["a"] and group.valid
        submit(dispatcher, records, "new", 3)
        done = dispatcher.run("new", execute, output, participates=True)
        assert done.result.completed_updates == 2 and calls == ["a", "new"]


def test_lost_abort_ack_is_uncertain_and_terminates_all_owned_workers():
    with workers(reject=lambda rank, command: rank == 1, break_abort=True) as (
        group,
        events,
        stopped,
    ):
        with pytest.raises(ProtocolError, match="absent"):
            group.run(commands())
        assert not group.valid and stopped == [True]
        assert all(phase == "stage" for _, phase, _ in events)
        with pytest.raises(RuntimeError, match="never replay"):
            group.run(commands(2, 2))


def test_metadata_failure_is_checked_for_all_workers_before_the_first_send():
    with workers() as (group, events, stopped):
        changed = commands()
        changed[1] = replace(changed[1], candidate="x" * 128, invocation=2)
        with pytest.raises(ExecutionNotStarted, match="idle worker"):
            group.run(changed)
        with pytest.raises(ExecutionNotStarted, match="participants"):
            group.run({0: changed[0]})
        assert not events and not stopped and group.valid
        group.run(commands())


def test_prepared_request_uses_the_same_group_and_actual_input_integrity_checks(
    tmp_path,
):
    with (
        prepared_invocation(tmp_path) as (data, transition, _, bindings, files),
        workers(
            count=1,
            binding=lambda rank, command: bindings.bind(
                command, lambda ref: files[ref.id]
            ),
        ) as (group, events, stopped),
    ):
        result = group.execute(data, transition)
        assert result[0].status == "completed"
        assert result[0].ticket.request_id == data.request.identity.request_id
        assert events == [(0, "execute", 1)]
        assert not stopped


def test_unavailable_connection_never_certifies_preserved_mutable_state():
    with workers() as (group, events, stopped):
        group.connections[1].valid = False
        with pytest.raises(RuntimeError, match="never replay") as failure:
            group.run(commands())
        assert not isinstance(failure.value, ExecutionNotStarted)
        assert stopped == [True] and not group.valid and not events


def test_group_closes_private_control_before_waiting_for_orderly_worker_exit():
    order = []
    connection = SimpleNamespace(
        scope=SCOPE, valid=True, close=lambda: order.append("eof")
    )
    group = WorkerGroup({0: connection}, stop=lambda: order.append("wait"))
    group.close()
    assert order == ["eof", "wait"]


def test_connection_rejects_wrong_tickets_or_phase_effects_before_reuse():
    command = call()
    for reply in (
        WorkerReply(
            replace(ticket(command), call_sha256="0" * 64), "ready", "not_started"
        ),
        WorkerReply(ticket(command), "completed", "completed"),
    ):
        incoming = io.BytesIO()
        write_message(incoming, reply)
        incoming.seek(0)
        connection = WorkerConnection(SCOPE, incoming, io.BytesIO())
        connection.stage(command)
        with pytest.raises(ProtocolError):
            connection.staged()
        assert not connection.valid
        with pytest.raises(RuntimeError, match="never resend"):
            connection.stage(command)
        connection.close()


@pytest.mark.parametrize("mode", ["normal", "lost_completion"])
def test_group_controls_actual_children_and_never_replays_a_lost_completion(
    tmp_path, mode
):
    children, connections = [], {}
    for rank in range(2):
        child = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).with_name("_worker_loop.py")),
                str(tmp_path / f"rank-{rank}"),
                mode if rank == 0 else "normal",
                str(rank),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        children.append(child)
        connections[rank] = WorkerConnection(
            replace(SCOPE, rank=rank), child.stdout, child.stdin
        )

    stopped = []

    def stop():
        stopped.append(True)
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
            child.stderr.close()

    with WorkerGroup(connections, stop=stop) as group:
        if mode == "normal":
            assert len(group.run(commands())) == 2
            assert [path.read_text() for path in sorted(tmp_path.iterdir())] == [
                "1\n",
                "1\n",
            ]
        else:
            with pytest.raises(ProtocolError, match="absent"):
                group.run(commands())
            assert not group.valid and stopped == [True]
            assert (tmp_path / "rank-0").read_text() == "1\n"
            with pytest.raises(RuntimeError, match="never replay"):
                group.run(commands(2, 2))
            assert (tmp_path / "rank-0").read_text() == "1\n"
    assert stopped == [True]
