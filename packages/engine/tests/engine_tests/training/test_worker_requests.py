"""File ownership follows confirmed effects and retained cleanup obligations."""

import io
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from engine_tests.training.test_worker_commands import invocation
from engine_tests.training.test_worker_output_frame import description, result
from shadowspill.orchestration.launch.file_grants import FileGrants, resolve_file
from shadowspill.orchestration.launch.publication import OutputFiles, require_output
from shadowspill.training.session.dispatch import ExecutionNotStarted
from shadowspill.training.worker.connection import WorkerConnection
from shadowspill.training.worker.group import WorkerGroup
from shadowspill.training.worker.output_files import FileOutputs
from shadowspill.training.worker.output_frame import WorkerOutputFrame
from shadowspill.training.worker.protocol import WorkerReply, ticket
from shadowspill.training.worker.requests import NodeFiles, WorkerRequests
from shadowspill_client.buffers import BufferBudget


@contextmanager
def configured(tmp_path, monkeypatch, *, failure=None, operation="step"):
    disk, memory, metadata = (
        BufferBudget(16 << 20, 100),
        BufferBudget(16 << 20, 100),
        BufferBudget(16 << 20),
    )
    with (
        invocation(tmp_path, operation) as (data, transition, command, bindings, _),
        FileGrants(root=tmp_path, memory=memory) as grants,
        OutputFiles(root=tmp_path, disk=disk, memory=memory) as outputs,
    ):
        selected = data.worker_plans(transition)[0]
        frames = (
            {}
            if selected is None
            else {
                (command.candidate, selected): WorkerOutputFrame(
                    *description(count=len(command.frames))
                )
            }
        )
        worker = SimpleNamespace(
            startup=SimpleNamespace(scope=bindings.scope),
            bindings=bindings,
            outputs=frames,
        )
        connection = WorkerConnection(bindings.scope, io.BytesIO(), io.BytesIO())
        connection.next_invocation = command.invocation
        stops, executions = [], []

        def stop():
            stops.append(True)
            if failure == "stop" and len(stops) == 1:
                raise OSError("stop failed")

        group = WorkerGroup({0: connection}, stop=stop)
        sink = FileOutputs(worker.outputs, root=outputs.root)

        def run(commands):
            received = commands[0]
            bound = bindings.bind(received, lambda ref: resolve_file(grants.root, ref))
            bound.verify_inputs(BufferBudget(16 << 20))
            sink.check(bound)
            if failure == "stage":
                raise ExecutionNotStarted("all workers aborted")
            executions.append(received)
            if failure in ("lost", "stop"):
                group.close()
                raise OSError("completion lost")
            if failure == "publication":
                return {
                    0: WorkerReply(
                        ticket(received), "failed", "completed", message="disk full"
                    )
                }
            ref = sink.collect(
                bound, result(count=len(received.frames)), BufferBudget(16 << 20)
            )
            return {
                0: WorkerReply(ticket(received), "completed", "completed", result=ref)
            }

        monkeypatch.setattr(group, "run", run)
        transport = WorkerRequests(
            group,
            {0: worker},
            {0: NodeFiles(tmp_path, disk, memory, grants, outputs)},
            metadata=metadata,
        )
        try:
            yield SimpleNamespace(
                data=data,
                transition=transition,
                transport=transport,
                group=group,
                outputs=outputs,
                grants=grants,
                disk=disk,
                memory=memory,
                metadata=metadata,
                stops=stops,
                executions=executions,
            )
        finally:
            transport.close()
        assert not transport._owners and metadata.used == disk.used == (0, 0)
    assert memory.used == (0, 0)


def test_results_retain_inputs_outputs_and_metadata_through_verification(
    tmp_path, monkeypatch
):
    with configured(tmp_path, monkeypatch) as ctx:
        work = ctx.transport.execute(ctx.data, ctx.transition)
        assert len(ctx.executions) == 1 and ctx.metadata.used[0] > 0
        assert ctx.grants._owners and ctx.disk.used[0] > 0
        with pytest.raises(ValueError, match="verified"), work.open(0):
            pass
        work.verify()
        with work.open(0) as stream:
            assert stream.read(8)
            with pytest.raises(RuntimeError, match="active producer/consumer"):
                work.close()
            assert ctx.metadata.used[0] > 0 and ctx.disk.used[0] > 0
        work.close()
        work.close()
        assert ctx.metadata.used == ctx.disk.used == (0, 0)
        assert ctx.group.valid


def test_discard_needs_command_metadata_but_no_tensor_file_owners(
    tmp_path, monkeypatch
):
    with configured(tmp_path, monkeypatch, operation="discard") as ctx:
        work = ctx.transport.execute(ctx.data, ctx.transition)
        assert len(ctx.executions) == 1 and ctx.metadata.used[0] > 0
        assert not ctx.grants._owners and ctx.disk.used == (0, 0)
        assert not work.frames and not work._deliveries and not work._files
        work.verify()
        assert work.replies[0].result is None
        work.close()
        assert ctx.metadata.used == (0, 0) and ctx.group.valid


@pytest.mark.parametrize("invalid", [False, True])
def test_close_confirms_group_stop_before_publishing_ready_work(
    tmp_path, monkeypatch, invalid
):
    with configured(tmp_path, monkeypatch, operation="close") as ctx:
        transition = ctx.transition
        if invalid:
            transition = replace(
                transition, before=replace(transition.before, valid=False)
            )
            ctx.group.valid = False
        work = ctx.transport.execute(ctx.data, transition)
        assert ctx.group.closed and ctx.transport.closed and ctx.stops == [True]
        assert len(ctx.executions) == (0 if invalid else 1)
        assert work.ready
        work.verify()
        assert ctx.metadata.used[0] > 0
        work.close()
        assert ctx.metadata.used == (0, 0)


def test_failed_close_keeps_cleanup_owned_until_explicit_retry(tmp_path, monkeypatch):
    with configured(tmp_path, monkeypatch, operation="close", failure="stop") as ctx:
        with pytest.raises(OSError, match="stop failed"):
            ctx.transport.execute(ctx.data, ctx.transition)
        assert not ctx.group.closed and ctx.metadata.used[0] > 0
        assert not ctx.group.valid
        transition = replace(
            ctx.transition, before=replace(ctx.transition.before, valid=False)
        )
        work = ctx.transport.execute(ctx.data, transition)
        assert ctx.group.closed and ctx.stops == [True, True]
        assert len(ctx.executions) == 1  # Retry stops the group; never sends execute.
        work.verify()
        work.close()
        ctx.transport.close()
        assert ctx.metadata.used == (0, 0)


@pytest.mark.parametrize("failure", ["stage", "lost", "stop"])
def test_abort_and_lost_worker_cleanup_have_distinct_file_lifetimes(
    tmp_path, monkeypatch, failure
):
    with configured(tmp_path, monkeypatch, failure=failure) as ctx:
        with pytest.raises((ExecutionNotStarted, OSError)):
            ctx.transport.execute(ctx.data, ctx.transition)
        if failure == "stop":
            assert not ctx.group.valid and not ctx.group.closed
            assert ctx.metadata.used[0] > 0 and ctx.disk.used[0] > 0
            assert ctx.grants._owners
            for work in ctx.transport._owners:
                with pytest.raises(RuntimeError, match="termination"):
                    work.close()
            ctx.transport.close()
            assert ctx.stops == [True, True]
        else:
            assert ctx.metadata.used == ctx.disk.used == (0, 0)
            assert ctx.group.valid == (failure == "stage")
        assert not ctx.transport._owners


@pytest.mark.parametrize("failure", ["publication", "checksum"])
def test_confirmed_publication_failure_is_deferred_to_the_publisher(
    tmp_path, monkeypatch, failure
):
    with configured(tmp_path, monkeypatch, failure=failure) as ctx:
        work = ctx.transport.execute(ctx.data, ctx.transition)
        assert work.ready and ctx.group.valid  # Dispatcher can record the effects.
        if failure == "checksum":
            path = require_output(ctx.outputs.root, work._files[0].spec)
            with path.open("r+b", buffering=0) as stream:
                stream.write(b"changed!")
        with pytest.raises((ValueError, OSError)):
            work.verify()
        work.close()
        assert ctx.group.valid and ctx.metadata.used == ctx.disk.used == (0, 0)


@pytest.mark.parametrize("failure", ["metadata", "source", "output"])
def test_all_file_preparation_finishes_before_any_worker_stage(
    tmp_path, monkeypatch, failure
):
    with configured(tmp_path, monkeypatch) as ctx:
        if failure == "metadata":
            ctx.transport._metadata = BufferBudget(1)
        elif failure == "source":
            next(iter(ctx.data.spool.frames.values())).path.unlink()
        else:

            def full(spec):
                raise OSError("output storage full")

            monkeypatch.setattr(ctx.outputs, "reserve", full)
        with pytest.raises(ExecutionNotStarted):
            ctx.transport.execute(ctx.data, ctx.transition)
        assert not ctx.executions and ctx.group.valid
        assert ctx.metadata.used == ctx.disk.used == (0, 0)


def test_failed_unpublished_cleanup_keeps_owner_for_close_retry(tmp_path, monkeypatch):
    with configured(tmp_path, monkeypatch) as ctx:

        def full(spec):
            raise OSError("output storage full")

        original = ctx.grants._release

        def denied(owner):
            raise OSError("unlink failed")

        with monkeypatch.context() as patch:
            patch.setattr(ctx.outputs, "reserve", full)
            patch.setattr(ctx.grants, "_release", denied)
            with pytest.raises(ExecutionNotStarted, match="cleanup retains"):
                ctx.transport.execute(ctx.data, ctx.transition)
            assert ctx.group.valid and ctx.transport._owners
            assert ctx.metadata.used[0] > 0 and ctx.disk.used[0] > 0
        assert ctx.grants._release == original
        ctx.transport.close()
        assert ctx.metadata.used == ctx.disk.used == (0, 0)


def test_shared_node_delivers_once_and_reserves_all_rank_outputs_before_release(
    tmp_path, monkeypatch
):
    with configured(tmp_path, monkeypatch) as ctx:
        original = ctx.transport.prepared[0]
        nodes = ctx.transport.nodes[0]
        commands = dict(ctx.group.commands(ctx.data, ctx.transition))
        scope = replace(original.startup.scope, rank=1)
        commands[1] = replace(commands[0], scope=scope)

        def bind(command, resolve):
            # This CPU test checks shared file ownership, not a distributed model.
            bound = original.bindings.bind(
                replace(command, scope=original.startup.scope), resolve
            )
            return replace(bound, command=command)

        workers = {
            **ctx.transport.prepared,
            1: SimpleNamespace(
                startup=SimpleNamespace(scope=scope),
                bindings=SimpleNamespace(bind=bind),
                outputs=original.outputs,
            ),
        }
        group = WorkerGroup(
            {
                rank: WorkerConnection(worker.startup.scope, io.BytesIO(), io.BytesIO())
                for rank, worker in workers.items()
            },
            stop=lambda: None,
        )
        monkeypatch.setattr(group, "commands", lambda *args: commands)
        metadata = BufferBudget(32 << 20)

        def run(received):
            assert len(list(ctx.outputs.root.glob("*.permit"))) == 2
            assert len(ctx.grants._owners) == 1
            replies = {}
            for rank, command in received.items():
                bound = bind(command, lambda ref: resolve_file(ctx.grants.root, ref))
                sink = FileOutputs(workers[rank].outputs, root=ctx.outputs.root)
                sink.check(bound)
                ref = sink.collect(
                    bound, result(count=len(command.frames)), BufferBudget(1 << 20)
                )
                replies[rank] = WorkerReply(
                    ticket(command), "completed", "completed", result=ref
                )
            return replies

        monkeypatch.setattr(group, "run", run)
        with WorkerRequests(
            group, workers, {0: nodes, 1: replace(nodes)}, metadata=metadata
        ) as transport:
            work = transport.execute(ctx.data, ctx.transition)
            assert len(work._deliveries) == len(work._grants) == 1
            work.verify()
            assert work.replies[0].result.id != work.replies[1].result.id
            work.close()
        assert metadata.used == ctx.disk.used == (0, 0)
