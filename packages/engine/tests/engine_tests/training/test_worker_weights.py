"""Exported model values enter fresh workers through verified, bounded startup."""

import hashlib
import io
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from engine_tests.checkpoint.test_conversion import WEIGHT, HostRank
from engine_tests.checkpoint.test_weights_load import HostExecutor, family
from engine_tests.training.test_worker_startup import prepared_worker
from shadowspill.checkpoint import write_weights
from shadowspill.checkpoint.identity import state_identity
from shadowspill.training.definitions import InitializationRuntime, ModelInitialization
from shadowspill.training.worker import runtime as module
from shadowspill.training.worker.channel import read_message, write_message
from shadowspill.training.worker.checkpoint_records import RankRestore, RankWeights
from shadowspill.training.worker.protocol import CODEC, WorkerBoot, WorkerLaunch
from shadowspill.training.worker.runtime import PreparedWorker
from shadowspill.training.worker.weights_reader import initialize_weights, validate_rank
from shadowspill_client import Weights
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.checkpoints import (
    CHECKPOINT_MEDIA_TYPE,
    RANK_MEDIA_TYPE,
    WEIGHTS_MEDIA_TYPE,
    WEIGHTS_RANK_MEDIA_TYPE,
    SavedRank,
    SaveInfo,
)
from shadowspill_client.protocol.frozen import FrozenBase
from shadowspill_client.protocol.requests import ArtifactRef


def source(tmp_path, *, partial=False, weights="master"):
    original = prepared_worker()
    setup = original.bindings.setup
    base = FrozenBase(setup.config.model, setup.model.source) if partial else None
    artifact = ArtifactRef("exported-weights", "c" * 64, 123, WEIGHTS_MEDIA_TYPE)
    selected = replace(setup.config.model, initialize=Weights.artifact(artifact.id))
    prepared = original.bindings.prepared
    prepared = replace(
        prepared,
        inventory=replace(
            prepared.inventory,
            layouts=replace(
                prepared.inventory.layouts,
                setup=replace(setup, config=replace(setup.config, model=selected)),
            ),
        ),
    )
    state = family(1, 0, frozen=True)[0]
    info = SaveInfo(
        "old-session",
        "old-run",
        "old-history",
        "export",
        9,
        1,
        original.startup.bundle,
        original.startup.bundle,
        sys.byteorder,
        weights=weights,
        frozen_base=base,
        frozen_state_id=None if base is None else base.identity,
        kind="weights",
    )
    path = tmp_path / "rank.ssp"
    with path.open("wb") as stream:
        write_weights(
            HostRank(state, initialize=True),
            stream,
            metadata=info.rank_metadata(),
            weights=weights,
            frozen_state_id=info.frozen_state_id,
            chunk_bytes=17,
        )
    rank = SavedRank(
        0,
        ArtifactRef(
            "weights-rank",
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_size,
            WEIGHTS_RANK_MEDIA_TYPE,
        ),
        state_identity(state),
    )
    initial = RankWeights(
        artifact, replace(artifact, id="converted-weights"), info, rank
    )
    worker = PreparedWorker.resolve(
        prepared,
        original.startup.scope,
        bundle=original.startup.bundle,
        initial_weights=initial,
    )
    return worker, state, path


def with_actual_plan(worker, state):
    return replace(
        worker,
        bindings=SimpleNamespace(
            primary="main",
            prepared=SimpleNamespace(plans={0: {"main": state.plan}}),
            setup=worker.bindings.setup,
            scope=worker.startup.scope,
        ),
    )


@pytest.mark.parametrize("partial", [False, True])
def test_exported_startup_retains_only_declared_base_provenance(tmp_path, partial):
    worker, _, _ = source(tmp_path, partial=partial)
    assert worker.restore is None
    assert worker.frozen_base == worker.initial_weights.info.frozen_base
    assert (worker.initialization is not None) == partial
    if partial:
        assert worker.initialization.context.weights.kind == "random"
    launch = WorkerLaunch(worker.startup, (), initial_weights=worker.initial_weights)
    assert CODEC.decode(CODEC.encode(launch)) == launch
    changed = replace(
        worker.initial_weights,
        generation=replace(
            worker.initial_weights.generation, id="different-generation"
        ),
    )
    other = PreparedWorker.resolve(
        worker.bindings.prepared,
        worker.startup.scope,
        bundle=worker.startup.bundle,
        initial_weights=changed,
    )
    assert other.startup.initialization_sha256 != worker.startup.initialization_sha256


def test_weights_startup_rejects_foreign_source_plans_rank_and_checkpoint(tmp_path):
    worker, _, _ = source(tmp_path)
    initial = worker.initial_weights
    for change in (
        {"artifact": replace(initial.artifact, id="different")},
        {"info": replace(initial.info, plans=replace(initial.info.plans, id="other"))},
        {"info": replace(initial.info, world_size=2)},
    ):
        with pytest.raises(ValueError, match=r"selected source or Plans|assignment"):
            PreparedWorker.resolve(
                worker.bindings.prepared,
                worker.startup.scope,
                bundle=worker.startup.bundle,
                initial_weights=replace(initial, **change),
            )
    for change in (
        {"artifact": replace(initial.artifact, media_type=CHECKPOINT_MEDIA_TYPE)},
        {"generation": replace(initial.generation, media_type=CHECKPOINT_MEDIA_TYPE)},
        {"info": replace(initial.info, kind="checkpoint")},
        {"rank": replace(initial.rank, rank=1)},
        {
            "rank": replace(
                initial.rank,
                archive=replace(initial.rank.archive, media_type=RANK_MEDIA_TYPE),
            )
        },
    ):
        with pytest.raises(ValueError, match="weights generation"):
            replace(initial, **change)
    restored = RankRestore(
        replace(initial.artifact, media_type=CHECKPOINT_MEDIA_TYPE),
        replace(initial.info, kind="checkpoint"),
        replace(
            initial.rank,
            archive=replace(initial.rank.archive, media_type=RANK_MEDIA_TYPE),
        ),
    )
    with pytest.raises(ValueError, match="cannot combine"):
        WorkerLaunch(worker.startup, (), restore=restored, initial_weights=initial)
    with pytest.raises(ValueError, match="cannot combine"):
        PreparedWorker.resolve(
            worker.bindings.prepared,
            worker.startup.scope,
            bundle=worker.startup.bundle,
            restore=restored,
            initial_weights=initial,
        )
    with pytest.raises(ValueError, match="exact frozen base"):
        WorkerLaunch(worker.startup, (), [initial.artifact], initial_weights=initial)


@pytest.mark.parametrize("fault", [None, "payload", "metadata", "state", "precision"])
def test_archive_verification_finishes_before_any_worker_startup_release(
    tmp_path, monkeypatch, fault
):
    worker, state, path = source(tmp_path)
    if fault == "payload":
        with path.open("r+b") as stream:
            stream.seek(512)
            stream.write(b"bad!")
    elif fault is not None:
        initial = worker.initial_weights
        if fault == "metadata":
            initial = replace(initial, info=replace(initial.info, completed_updates=10))
        elif fault == "state":
            initial = replace(
                initial, rank=replace(initial.rank, state_sha256="0" * 64)
            )
        else:
            initial = replace(initial, info=replace(initial.info, weights="compute"))
        worker = replace(worker, initial_weights=initial)
    worker = with_actual_plan(worker, state)
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
    def opened(self, *args, **kwargs):
        assert read_message(io.BytesIO(outgoing.getvalue())) == WorkerBoot(
            worker.startup, "prepared"
        )
        events.append("open")
        yield SimpleNamespace(
            loop=SimpleNamespace(serve=lambda *a: events.append("serve"))
        )

    monkeypatch.setattr(PreparedWorker, "open", opened)

    def serve():
        worker.serve(
            incoming,
            outgoing,
            InitializationRuntime(tmp_path, lambda _: pytest.fail("fresh source")),
            lambda _: path,
            None,
        )

    if fault:
        with pytest.raises(ValueError):
            serve()
        assert not events and outgoing.sent == b""
    else:
        serve()
        assert events == ["open", "serve"]
    assert incoming.closed and outgoing.closed


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("weights", ["master", "compute"])
def test_worker_weights_loader_retains_model_values_and_initializes_fresh_defaults(
    tmp_path, partial, weights
):
    worker, state, path = source(tmp_path, partial=partial, weights=weights)
    budget = BufferBudget(66)
    events = []

    @contextmanager
    def open_base(runtime):
        events.append("open")
        assert budget.used == (66, 1)

        def fill(chunk, destination):
            assert chunk.name == "base.weight"
            destination.copy_(
                WEIGHT.flatten()[chunk.offset : chunk.offset + destination.numel()]
            )

        try:
            yield fill
        finally:
            events.append("close")

    initial = (
        None
        if not partial
        else replace(
            worker.initialization,
            description=ModelInitialization(
                peak_bytes=32, chunk_bytes=17, open=open_base
            ),
        )
    )
    executor = HostExecutor(state)
    executor.controls.host["optimizer.linear.weight.step"].fill_(100)
    runtime = InitializationRuntime(
        tmp_path, lambda _: pytest.fail("unexpected source")
    )
    validate_rank(state, path, worker.initial_weights, chunk_bytes=17, world_size=1)
    initialize_weights(
        executor, path, worker.initial_weights, initial, runtime, budget, chunk_bytes=17
    )
    expected = WEIGHT if weights == "master" else WEIGHT.bfloat16().float()
    torch.testing.assert_close(
        executor.snapshot()["master.linear.weight"], expected.flatten(), rtol=0, atol=0
    )
    assert executor.controls.host["optimizer.linear.weight.step"].item() == 0
    assert executor.controls.host["optimizer_group.0.lr"].item() == pytest.approx(0.04)
    assert not executor.values["moment"].count_nonzero()
    assert not executor.values["variance"].count_nonzero()
    assert worker.initial_weights.info.completed_updates == 9
    assert events == (["open", "close"] if partial else [])
    assert budget.used == (0, 0)


def test_worker_open_uses_fresh_randomness_and_preserves_full_export_provenance(
    tmp_path, monkeypatch
):
    worker, state, path = source(tmp_path)
    worker = with_actual_plan(worker, state)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        module, "workers", lambda *a, **kw: nullcontext(SimpleNamespace(logical_rank=0))
    )
    captures = []
    for _ in range(2):
        executor = HostExecutor(state)
        executor.device = torch.device("cpu")
        executor.spill_capacity_bytes = worker.startup.spill_bytes
        monkeypatch.setattr(
            "shadowspill.execution.executor.Executor",
            lambda *a, executor=executor, **kw: nullcontext(executor),
        )
        with worker.open(
            InitializationRuntime(tmp_path, lambda _: pytest.fail("fresh source")),
            lambda _: path,
            SimpleNamespace(collect=lambda *a: None),
        ) as running:
            assert executor._initialized
            assert executor.controls.host["optimizer.linear.weight.step"].item() == 0
            execution = running.loop._execute
            assert execution._restore.frozen_base is None
            assert (
                execution._randomness.root_seed
                == worker.bindings.setup.config.execution.seed
            )
            captures.append(execution._randomness.capture())
            torch.rand(31)
        assert running.staging.used == (0, 0)
    assert captures[0] == captures[1]
