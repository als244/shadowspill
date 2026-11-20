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
from shadowspill.checkpoint.random_streams import (
    RandomState,
    default_streams,
    snapshot,
)
from shadowspill.checkpoint.state import PlanState
from shadowspill.training.components import InitializationRuntime
from shadowspill.training.worker.channel import read_message, write_message
from shadowspill.training.worker.checkpoint_reader import WorkerCheckpointReader
from shadowspill.training.worker.checkpoint_records import RankRestore
from shadowspill.training.worker.protocol import CODEC, WorkerBoot, WorkerLaunch
from shadowspill.training.worker.runtime import PreparedWorker
from shadowspill_client import Weights
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.checkpoints import (
    CHECKPOINT_MEDIA_TYPE,
    RANK_MEDIA_TYPE,
    SavedRank,
    SaveInfo,
)
from shadowspill_client.protocol.frozen import FrozenBase
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
    info = SaveInfo(
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
    save_checkpoint(
        executor,
        path,
        metadata={
            **info.rank_metadata(),
            "randomness": snapshot(
                0,
                [
                    RandomState(stream, seed=1)
                    for stream in default_streams(
                        0, torch.__version__, accelerator=True
                    )
                ],
            ),
        },
        chunk_bytes=16,
    )
    ref = ArtifactRef(
        "rank",
        hashlib.sha256(path.read_bytes()).hexdigest(),
        path.stat().st_size,
        RANK_MEDIA_TYPE,
    )
    restore = RankRestore(
        ArtifactRef("checkpoint", "b" * 64, 123, CHECKPOINT_MEDIA_TYPE),
        info,
        SavedRank(0, ref, state_identity(executor)),
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


def test_checkpoint_launch_grants_only_its_exact_artifact_backed_frozen_base(tmp_path):
    worker, _, _ = source(tmp_path)
    config = prepared_worker().bindings.setup.config
    artifact = ArtifactRef("frozen-base", "c" * 64, 100)
    base = FrozenBase(
        replace(config.model, initialize=Weights.artifact(artifact.id)),
        config.component_sources["model"],
        artifact,
    )
    restored = replace(
        worker.restore,
        info=replace(
            worker.restore.info, frozen_base=base, frozen_state_id=base.identity
        ),
    )
    launch = WorkerLaunch(worker.startup, (), [artifact], restored)
    assert CODEC.decode(CODEC.encode(launch)) == launch
    other = ArtifactRef("unrelated", "d" * 64, 100)
    for grants in ((), [other], [artifact, other]):
        with pytest.raises(ValueError, match="frozen base"):
            WorkerLaunch(worker.startup, (), grants, restored)
