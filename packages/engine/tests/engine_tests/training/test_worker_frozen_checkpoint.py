"""Service rank saves reload only immutable base weights through installed loaders."""

import io
import json
import sys
import tarfile
from contextlib import contextmanager
from dataclasses import replace

import pytest
import torch

from engine_tests.execution.test_partial_checkpoint import executor
from engine_tests.training.test_worker_checkpoints import save_call
from engine_tests.training.test_worker_initialization import preparation
from engine_tests.training.test_worker_loop import SCOPE
from shadowspill.training.definitions import InitializationRuntime, ModelInitialization
from shadowspill.training.session.controls import CallControls
from shadowspill.training.worker.binary import receive_stream
from shadowspill.training.worker.bindings import BoundWorkerCall
from shadowspill.training.worker.checkpoint_reader import WorkerCheckpointReader
from shadowspill.training.worker.checkpoint_records import RankRestore
from shadowspill.training.worker.checkpoint_writer import WorkerCheckpointWriter
from shadowspill.training.worker.protocol import WorkerCall
from shadowspill_client import Weights
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.checkpoint_settings import CheckpointSave
from shadowspill_client.protocol.checkpoints import (
    CHECKPOINT_MEDIA_TYPE,
    CheckpointInfo,
)
from shadowspill_client.protocol.frozen import FrozenBase
from shadowspill_client.protocol.requests import ArtifactRef


def fixture(tmp_path, *, artifact=False, capacity=256, peak=128, fail=False):
    value, files, events = executor(), {}, []
    value.plan.program.digest = "a" * 64
    value.synchronize = lambda: None
    budget = BufferBudget(capacity)

    def describe(config, context):
        @contextmanager
        def open_loader(runtime):
            assert budget.used == (peak + 24, 1)
            events.append("open")
            try:
                if context.weights.kind == "artifact":
                    offset = int(
                        runtime.artifact(context.weights.reference).read_text()
                    )
                else:
                    offset = (context.weights.seed - 17) * 100

                def fill(chunk, target):
                    assert chunk.name == "base" and target.nbytes <= 12
                    events.append((chunk.name, chunk.offset, target.numel()))
                    target.copy_(
                        torch.arange(chunk.offset, chunk.offset + target.numel())
                        + offset
                    )
                    if fail:
                        raise OSError("base loader failed")

                yield fill
            finally:
                assert budget.used == (peak + 24, 1)
                events.append("close")

        return ModelInitialization(peak_bytes=peak, chunk_bytes=16, open=open_loader)

    setup = preparation(describe)
    base = FrozenBase(setup.config.model, setup.model.source)
    if artifact:
        from hashlib import sha256

        path = tmp_path / "weights"
        path.write_text("0")
        reference = ArtifactRef("weights", sha256(b"0").hexdigest(), 1)
        files[reference] = path
        base = replace(
            base,
            model=replace(base.model, initialize=Weights.artifact(reference.id)),
            artifact=reference,
        )
    reader = WorkerCheckpointReader(
        value,
        budget,
        files.__getitem__,
        chunk_bytes=12,
        world_size=1,
        setup=setup,
        runtime=InitializationRuntime(
            tmp_path, lambda _: pytest.fail("historical weights used")
        ),
        frozen_base=base,
    )
    return value, budget, base, reader, files, events


def info(base, *, request="save"):
    plans = ArtifactRef("plans", "b" * 64, 1)
    return CheckpointInfo(
        "session",
        "run",
        "history",
        request,
        1,
        1,
        plans,
        plans,
        sys.byteorder,
        frozen_base=base,
        frozen_state_id=None if base is None else base.identity,
    )


def save(tmp_path, reader, boundary, files):
    wire = io.BytesIO()
    writer = WorkerCheckpointWriter(
        reader.executor,
        reader.budget,
        wire,
        plans=boundary.plans,
        chunk_bytes=12,
        frozen_base=lambda: reader.frozen_base,
    )
    request = save_call().request
    request = replace(
        request,
        identity=replace(request.identity, request_id=boundary.request_id),
        save=CheckpointSave(metadata=boundary.metadata, weights=boundary.weights),
    )
    command = WorkerCall(SCOPE, 1, request, None, False, False, checkpoint=boundary)
    bound = writer.prepare(BoundWorkerCall(command, None, CallControls(), {}, ()))
    saved = writer(bound)
    wire.seek(0)
    path = tmp_path / (boundary.request_id + ".ssp")
    with path.open("wb") as destination:
        receive_stream(wire, destination, size_bytes=bound.checkpoint.size_bytes)
    row = bound.checkpoint.reference(boundary.request_id + ".rank", saved.sha256)
    files[row.archive] = path
    assert path.stat().st_size == row.archive.size_bytes
    with tarfile.open(path) as archive:
        manifest = json.loads(archive.extractfile("manifest.json").read())
    source = RankRestore(
        ArtifactRef(boundary.request_id, "c" * 64, 1, CHECKPOINT_MEDIA_TYPE),
        boundary,
        row,
    )
    return source, manifest


@pytest.mark.parametrize("artifact", [False, True])
def test_partial_service_rank_roundtrip_retains_mutations_buffers_and_optimizer(
    tmp_path, artifact
):
    value, budget, base, reader, files, events = fixture(tmp_path, artifact=artifact)
    expected = {name: tensor.clone() for name, tensor in value.values.items()}
    source, manifest = save(tmp_path, reader, info(base), files)
    assert manifest["omitted_parameters"] == ["base", "tied_base"]
    assert {row["name"] for row in manifest["tensors"]} == expected.keys() - {"base"}
    for tensor in value.values.values():
        tensor.fill_(-1)
    path = reader.prepare_source(source)
    assert not events  # Preflight describes the loader without opening it.
    reader.load(source, path)
    torch.testing.assert_close(value.values, expected, rtol=0, atol=0)
    assert events == ["open", ("base", 0, 3), ("base", 3, 3), "close"]
    assert reader.frozen_base == base and budget.used == (0, 0)


def test_restore_switches_base_and_full_restore_clears_omission_provenance(tmp_path):
    value, budget, base, reader, files, events = fixture(tmp_path)
    original, _ = save(tmp_path, reader, info(base, request="original"), files)
    newer = replace(base, model=replace(base.model, initialize=Weights.random(seed=18)))
    value.values["base"].add_(100)
    reader.frozen_base = newer  # A separately initialized source worker.
    source, _ = save(tmp_path, reader, info(newer, request="new-base"), files)
    reader.load(original, reader.prepare_source(original))
    assert reader.frozen_base == base
    reader.load(source, reader.prepare_source(source))
    assert reader.frozen_base == newer
    torch.testing.assert_close(value.values["base"], torch.arange(6).float() + 100)
    with pytest.raises(ValueError, match="differs from the initialized worker"):
        save(tmp_path, reader, info(base, request="stale"), files)
    saved, _ = save(tmp_path, reader, info(newer, request="following"), files)
    assert saved.info.frozen_base == newer
    reader.frozen_base = None
    value.values["base"].add_(25)
    complete, manifest = save(tmp_path, reader, info(None, request="full"), files)
    assert not manifest["omitted_parameters"]
    reader.load(original, reader.prepare_source(original))
    events.clear()
    reader.load(complete, reader.prepare_source(complete))
    assert reader.frozen_base is None and not events
    torch.testing.assert_close(value.values["base"], torch.arange(6).float() + 125)
    _, manifest = save(tmp_path, reader, info(None, request="after-full"), files)
    assert not manifest["omitted_parameters"] and budget.used == (0, 0)


@pytest.mark.parametrize(
    "failure", ["source", "model", "staging", "missing_artifact", "no_hook"]
)
def test_base_preflight_rejects_before_opening_or_mutating(tmp_path, failure):
    value, budget, base, reader, files, events = fixture(
        tmp_path, artifact=failure == "missing_artifact"
    )
    source, _ = save(tmp_path, reader, info(base), files)
    expected = {name: tensor.clone() for name, tensor in value.values.items()}
    if failure == "source":
        base = replace(base, source=replace(base.source, digest="f" * 64))
    elif failure == "model":
        base = replace(base, model=replace(base.model, config={"wrong": True}))
    elif failure == "staging":
        reader.budget = BufferBudget(128)
    elif failure == "missing_artifact":
        files.pop(base.artifact)
    else:
        reader.setup = None
    source = replace(
        source,
        info=replace(source.info, frozen_base=base, frozen_state_id=base.identity),
    )
    with pytest.raises((ValueError, KeyError)):
        reader.prepare_source(source)
    torch.testing.assert_close(value.values, expected, rtol=0, atol=0)
    assert not events and budget.used == reader.budget.used == (0, 0)


def test_failed_loader_never_confirms_new_base_and_releases_its_buffers(tmp_path):
    value, budget, base, reader, files, events = fixture(tmp_path, fail=True)
    source, _ = save(tmp_path, reader, info(base), files)
    reader.frozen_base = None
    with pytest.raises(OSError, match="base loader failed"):
        reader.load(source, reader.prepare_source(source))
    assert reader.frozen_base is None and not value.initialized
    assert events[0] == "open" and events[-1] == "close"
    assert budget.used == (0, 0)
