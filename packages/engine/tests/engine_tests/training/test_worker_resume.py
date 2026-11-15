"""Actual archive preflight completes before any worker startup release."""

import hashlib
import io
import sys
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from engine_tests.checkpoint.test_state import plan
from engine_tests.execution.test_checkpoint import MemoryExecutor
from engine_tests.offline.test_state_inventory import training
from engine_tests.training.test_worker_startup import prepared_worker
from shadowspill.checkpoint import save_checkpoint
from shadowspill.checkpoint.identity import state_identity
from shadowspill.checkpoint.state import PlanState
from shadowspill.training.definitions import InitializationRuntime
from shadowspill.training.worker.channel import read_message, write_message
from shadowspill.training.worker.checkpoint_reader import WorkerCheckpointReader
from shadowspill.training.worker.checkpoint_records import RankRestore
from shadowspill.training.worker.protocol import WorkerBoot, WorkerLaunch
from shadowspill.training.worker.runtime import PreparedWorker
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.checkpoints import (
    CHECKPOINT_MEDIA_TYPE,
    RANK_MEDIA_TYPE,
    CheckpointInfo,
    CheckpointRank,
)
from shadowspill_client.protocol.requests import ArtifactRef


def source(tmp_path, *, corrupt=False):
    worker = prepared_worker()
    target = plan(training())
    state = PlanState(target, {}, logical_rank=0)
    executor = MemoryExecutor()
    executor.plan = target
    executor.values = {
        chunk.name: torch.empty_strided(
            chunk.shape, chunk.stride, dtype=chunk.dtype
        ).fill_(3)
        for chunk in state.tensor_chunks()
    }
    if corrupt:
        executor.values["linear.weight"] = torch.ones(1)
    info = CheckpointInfo(
        "source-session",
        "source-run",
        "source-history",
        "save",
        9,
        1,
        worker.startup.bundle,
        worker.startup.bundle,
        sys.byteorder,
    )
    path = tmp_path / "rank.ssp"
    save_checkpoint(executor, path, metadata=info.rank_metadata(), chunk_bytes=16)
    ref = ArtifactRef(
        "rank",
        hashlib.sha256(path.read_bytes()).hexdigest(),
        path.stat().st_size,
        RANK_MEDIA_TYPE,
    )
    restore = RankRestore(
        ArtifactRef("checkpoint", "b" * 64, 123, CHECKPOINT_MEDIA_TYPE),
        info,
        CheckpointRank(0, ref, state_identity(executor)),
    )
    worker = replace(
        worker,
        initialization=None,
        restore=restore,
        bindings=SimpleNamespace(
            primary="main", prepared=SimpleNamespace(plans={0: {"main": target}})
        ),
    )
    return worker, executor, path


@pytest.mark.parametrize("corrupt", [False, True])
def test_complete_archive_geometry_is_checked_before_startup_prepared(
    tmp_path, monkeypatch, corrupt
):
    worker, _executor, path = source(tmp_path, corrupt=corrupt)
    events = []

    class Stream(io.BytesIO):
        def close(self):
            if not self.closed:
                self.sent = self.getvalue()
            super().close()

    incoming, outgoing = Stream(), Stream()
    write_message(incoming, WorkerBoot(worker.startup, "start"))
    incoming.seek(0)

    @contextmanager
    def opened(self, runtime, resolve, outputs, **options):
        events.append("open")
        assert read_message(io.BytesIO(outgoing.getvalue())) == WorkerBoot(
            worker.startup, "prepared"
        )
        yield SimpleNamespace(
            loop=SimpleNamespace(serve=lambda *a: events.append("serve"))
        )

    monkeypatch.setattr(PreparedWorker, "open", opened)

    def serve():
        worker.serve(
            incoming,
            outgoing,
            InitializationRuntime(tmp_path, lambda _: pytest.fail("fresh weights")),
            lambda _: path,
            None,
        )

    if corrupt:
        with pytest.raises(ValueError, match="state identity differs"):
            serve()
        assert not events and outgoing.sent == b""
    else:
        serve()
        assert events == ["open", "serve"]
    assert incoming.closed and outgoing.closed


def test_same_reader_initializes_checkpoint_without_a_training_request(tmp_path):
    worker, executor, path = source(tmp_path)
    expected = {name: value.clone() for name, value in executor.values.items()}
    for value in executor.values.values():
        value.zero_()
    executor.initialized = False
    executor.synchronize = lambda: None
    budget = BufferBudget(32)
    reader = WorkerCheckpointReader(
        executor, budget, lambda _: path, chunk_bytes=16, world_size=1
    )
    reader.load(worker.restore, reader.prepare_source(worker.restore))
    assert executor.initialized and budget.used == (0, 0)
    for name, value in executor.values.items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


def test_checkpoint_launch_cannot_also_grant_fresh_weights_or_another_rank(tmp_path):
    worker, _, _ = source(tmp_path)
    WorkerLaunch(worker.startup, (), restore=worker.restore)
    with pytest.raises(ValueError, match="no fresh weights"):
        WorkerLaunch(
            worker.startup, (), [ArtifactRef("weights", "c" * 64, 1)], worker.restore
        )
    with pytest.raises(ValueError, match="assignment"):
        PreparedWorker.resolve(
            prepared_worker().bindings.prepared,
            worker.startup.scope,
            bundle=worker.startup.bundle,
            restore=replace(
                worker.restore, info=replace(worker.restore.info, world_size=2)
            ),
        )
