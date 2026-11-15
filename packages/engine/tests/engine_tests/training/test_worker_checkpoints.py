"""Ordered saves use actual worker processes and the existing dispatcher/store."""

import io
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from engine_tests.execution.test_checkpoint import MemoryExecutor
from engine_tests.server.test_artifacts import reference, store
from engine_tests.server.test_request_records import request
from engine_tests.server.test_request_records import store as records_store
from shadowspill.checkpoint import load_checkpoint
from shadowspill.orchestration.launch.file_grants import FileGrants
from shadowspill.orchestration.launch.publication import OutputFiles
from shadowspill.server.checkpoints import RunCheckpoints
from shadowspill.server.results import ResultPublisher
from shadowspill.training.session.controls import CallControls
from shadowspill.training.session.dispatch import RequestDispatcher
from shadowspill.training.session.prepared import PreparedCall
from shadowspill.training.session.state import OperationState
from shadowspill.training.worker.bindings import BoundWorkerCall
from shadowspill.training.worker.connection import WorkerConnection
from shadowspill.training.worker.group import WorkerGroup
from shadowspill.training.worker.protocol import WorkerScope
from shadowspill.training.worker.requests import NodeFiles, WorkerRequests
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.checkpoint_settings import SaveSettings
from shadowspill_client.protocol.checkpoints import CheckpointManifest
from shadowspill_client.protocol.codec import decode, encode
from shadowspill_client.protocol.requests import OperationRequest


@contextmanager
def running(root, artifacts, *, mode="normal", count=2, weights=False):
    children, connections, prepared = [], {}, {}
    metadata, memory, disk = (BufferBudget(32 << 20, 10) for _ in range(3))
    grants = FileGrants(root=root, memory=memory)

    def stop():
        for child in children:
            if child.poll() is None:
                child.kill()
        for child in children:
            child.wait(timeout=10)

    def bind(command, resolve):
        return BoundWorkerCall(command, None, CallControls(), {}, ())

    try:
        for rank in range(count):
            child = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).with_name("_worker_checkpoint.py")),
                    str(rank),
                    str(root / f"executions-{rank}"),
                    mode if rank == 1 or mode == "frozen" else "normal",
                    str(grants.root),
                    str(count),
                    "weights" if weights else "checkpoint",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
            children.append(child)
            scope = WorkerScope("run", "server", f"process-{rank}", rank)
            connections[rank] = WorkerConnection(scope, child.stdout, child.stdin)
            if weights:
                from engine_tests.checkpoint.test_conversion import (
                    family as state_family,
                )

                program = state_family(count, 0)[rank].plan.program
            else:
                program = SimpleNamespace(digest="a" * 64)
            prepared[rank] = SimpleNamespace(
                startup=SimpleNamespace(scope=scope),
                outputs={},
                bindings=SimpleNamespace(
                    bind=bind,
                    primary="main",
                    prepared=SimpleNamespace(
                        plans={rank: {"main": SimpleNamespace(program=program)}}
                    ),
                ),
            )
        from engine_tests.training.test_prepared_requests import family

        config_body = encode(family().inventory.layouts.setup.config)
        config = reference(config_body, "config")
        artifacts.put(config, lambda _: io.BytesIO(config_body))
        plans = reference(b"plans", "plans")
        artifacts.put(plans, lambda _: io.BytesIO(b"plans"))
        checkpoints = RunCheckpoints(
            artifacts,
            session_id="session",
            configuration=config,
            plans=plans,
            world_size=count,
        )
        with (
            WorkerGroup(connections, stop=stop) as group,
            OutputFiles(root=root, disk=disk, memory=memory) as outputs,
            WorkerRequests(
                group,
                prepared,
                {
                    rank: NodeFiles(root, disk, memory, grants, outputs)
                    for rank in prepared
                },
                metadata=metadata,
                checkpoints=checkpoints,
            ) as workers,
            ResultPublisher(
                artifacts,
                session_id="session",
                root=root,
                memory=memory,
                disk=disk,
            ) as publisher,
        ):
            yield workers, publisher, checkpoints
        grants.close()
        assert metadata.used == memory.used == disk.used == (0, 0)
    finally:
        stop()
        grants.close()
        for child in children:
            for stream in (child.stdin, child.stdout, child.stderr):
                stream.close()


def save_call(sequence=1):
    command = OperationRequest(
        replace(request().identity, request_id=f"save-{sequence}", sequence=sequence),
        "save",
        save=SaveSettings(name="latest", metadata={"cursor": [2, 7]}),
    )
    return PreparedCall(
        command, None, None, CallControls(), None, BufferBudget(1), ranks=(0, 1)
    )


def dispatch(records, dispatcher, workers, publisher, call):
    records.submit(call.request)
    dispatcher.order.validated(call.request.identity.request_id)
    return dispatcher.run(
        call.request.identity.request_id,
        lambda transition: workers.execute(call, transition),
        lambda receipt, work: publisher(call, receipt, work),
    )


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "capacity",
        "no_encoder",
        "lost_completion",
        "receipt",
        "publication",
        "assembly",
    ],
)
def test_ordered_rank_save_admission_receipt_and_publication(
    tmp_path, monkeypatch, failure
):
    with (
        store(tmp_path) as artifacts,
        records_store(tmp_path) as records,
        running(tmp_path, artifacts, mode=failure or "normal") as (
            workers,
            publisher,
            checkpoints,
        ),
    ):
        call, state = save_call(), OperationState(completed_updates=9)
        dispatcher = RequestDispatcher(records, state, max_pending=4)
        before = artifacts.used
        if failure == "capacity":
            artifacts.max_bytes = before[0] + 1
        if failure == "receipt":

            def fail_receipt(receipt):
                raise OSError("receipt failed")

            monkeypatch.setattr(records, "executed", fail_receipt)
        if failure == "assembly":
            stage = checkpoints.stage

            def fail_assembly(info, ranks, write, *, name):
                def produce(rank, sink):
                    saved = write(rank, sink)
                    if rank.rank == 1:
                        raise OSError("manifest assembly failed after confirmations")
                    return saved

                return stage(info, ranks, produce, name=name)

            monkeypatch.setattr(checkpoints, "stage", fail_assembly)
        original = artifacts._commit

        def commit(*args):
            assert records.execution(call.request.identity.request_id) is not None
            if failure == "publication":
                raise OSError("publication failed")
            return original(*args)

        monkeypatch.setattr(artifacts, "_commit", commit)
        if failure == "receipt":
            with pytest.raises(OSError, match="receipt failed"):
                dispatch(records, dispatcher, workers, publisher, call)
            assert not state.boundary.valid
        else:
            outcome = dispatch(records, dispatcher, workers, publisher, call)
            assert outcome.state == ("completed" if failure is None else "failed"), (
                outcome
            )
        if failure is None:
            manifest_ref = outcome.result.checkpoint
            assert manifest_ref == artifacts.resolve_alias("session", "run", "latest")
            with artifacts.open(manifest_ref) as source:
                manifest = decode(source.read())
            assert isinstance(manifest, CheckpointManifest)
            assert (
                manifest.info.history_id
                == outcome.result.history_id
                == state.boundary.history_id
            )
            assert manifest.info.completed_updates == 9 and manifest.info.metadata[
                "cursor"
            ] == (2, 7)
            for row in manifest.ranks:
                restored = MemoryExecutor()
                restored.plan.program.digest = "a" * 64
                restored.logical_rank = row.rank
                expected = {
                    key: value.clone() for key, value in restored.values.items()
                }
                for tensor in restored.values.values():
                    tensor.zero_()
                path = tmp_path / f"restore-{row.rank}.ssp"
                with artifacts.open(row.archive) as source:
                    path.write_bytes(source.read())
                load_checkpoint(restored, path, chunk_bytes=24)
                for name, tensor in restored.values.items():
                    torch.testing.assert_close(tensor.float(), expected[name].float())
            assert state.boundary.completed_updates == 9 and state.boundary.valid
        else:
            assert artifacts.used == before
            assert artifacts.resolve_alias("session", "run", "latest") is None
            if failure in ("capacity", "no_encoder"):
                assert state.boundary.valid and workers.group.valid
                assert not list(tmp_path.glob("executions-*"))
            elif failure != "receipt":
                assert outcome.failure.execution_effect == (
                    "completed" if failure in ("publication", "assembly") else "unknown"
                )
        assert not workers._owners
        if failure in ("publication", "assembly"):
            assert state.boundary.valid and workers.group.valid
        if failure not in ("capacity", "no_encoder"):
            assert [
                (tmp_path / f"executions-{rank}").read_text() for rank in range(2)
            ] == ["1\n", "1\n"]


def test_save_rejects_open_accumulation_before_contacting_workers(tmp_path):
    with records_store(tmp_path) as records:
        state = OperationState()
        state.complete(state.begin("forward_backward", "previous", participates=True))
        dispatcher = RequestDispatcher(records, state, max_pending=4)
        call = save_call()

        def uncalled(*args):
            raise AssertionError("open accumulation released workers")

        result = dispatch(
            records, dispatcher, SimpleNamespace(execute=uncalled), uncalled, call
        )
        assert result.failure.code == "operation_conflict"
        assert result.failure.execution_effect == "not_started"
        assert state.boundary.valid and state.boundary.accumulation.open


def test_corrected_save_after_capacity_rejection_executes_once(tmp_path):
    with (
        store(tmp_path) as artifacts,
        records_store(tmp_path) as records,
        running(tmp_path, artifacts) as (workers, publisher, _),
    ):
        state = OperationState(completed_updates=9)
        dispatcher = RequestDispatcher(records, state, max_pending=4)
        capacity = artifacts.max_bytes
        artifacts.max_bytes = artifacts.used[0] + 1
        failed = dispatch(records, dispatcher, workers, publisher, save_call())
        assert failed.failure.execution_effect == "not_started"
        artifacts.max_bytes = capacity
        complete = dispatch(records, dispatcher, workers, publisher, save_call(2))
        assert complete.state == "completed" and complete.result.completed_updates == 9
        assert [(tmp_path / f"executions-{rank}").read_text() for rank in range(2)] == [
            "2\n",
            "2\n",
        ]


def test_completion_envelope_is_checked_before_worker_release():
    from shadowspill.training.worker.checkpoint_records import RankSave
    from shadowspill.training.worker.loop import WorkerLoop
    from shadowspill.training.worker.protocol import CODEC, WorkerCall
    from shadowspill_client.protocol.checkpoints import SaveInfo
    from shadowspill_client.protocol.requests import ArtifactRef

    call = save_call()
    command = replace(
        call.request, save=SaveSettings(metadata={"large": "x" * 450_000})
    )
    info = SaveInfo(
        "session",
        "run",
        "history",
        command.identity.request_id,
        9,
        2,
        ArtifactRef("config", "a" * 64, 3),
        ArtifactRef("plans", "b" * 64, 5),
        sys.byteorder,
        metadata=command.save.metadata,
    )
    scope = WorkerScope("run", "server", "process-0", 0)
    wire = WorkerCall(scope, 1, command, None, False, False, checkpoint=info)
    CODEC.encode(wire)
    description = RankSave(
        0, 1024, "c" * 64, "d" * 64, {f"p{i:04d}": "e" * 64 for i in range(8000)}
    )
    bound = BoundWorkerCall(wire, None, CallControls(), {}, (), checkpoint=description)

    def uncalled(*args):
        raise AssertionError("oversized completion released execution")

    loop = WorkerLoop(scope, lambda _: bound, uncalled, BufferBudget(1))
    reply = loop.handle(wire)
    assert reply.status == "failed" and reply.effect == "not_started"
    assert "exceeds" in reply.message and loop.valid
