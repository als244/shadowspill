"""Real CPU updates through worker execution and its native checkpoint stream."""

import io
import sys
from types import SimpleNamespace

import pytest
import torch

from engine_tests.execution.test_checkpoint import MemoryExecutor
from engine_tests.training.test_worker_checkpoints import save_call
from engine_tests.training.test_worker_loop import SCOPE
from engine_tests.training.test_worker_randomness import draw, owner
from shadowspill.checkpoint import save_checkpoint
from shadowspill.pytorch._rng import preserve_rng
from shadowspill.training.session.controls import CallControls
from shadowspill.training.worker.binary import receive_stream
from shadowspill.training.worker.bindings import BoundWorkerCall
from shadowspill.training.worker.checkpoint_reader import WorkerCheckpointReader
from shadowspill.training.worker.checkpoint_records import RankRestore
from shadowspill.training.worker.checkpoint_writer import WorkerCheckpointWriter
from shadowspill.training.worker.execution import WorkerExecution
from shadowspill.training.worker.protocol import WorkerCall
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.checkpoints import (
    CHECKPOINT_MEDIA_TYPE,
    CheckpointInfo,
)
from shadowspill_client.protocol.requests import ArtifactRef


@pytest.fixture(autouse=True)
def preserved():
    with preserve_rng(torch.device("cpu")):
        yield


def runtime():
    streams = owner()
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0, 3.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    executor = MemoryExecutor()
    executor.plan.program.digest = "a" * 64
    executor.values = {"parameter": parameter.detach()}
    executor.synchronize = lambda: None

    def execute(*, plan, **kwargs):
        if plan != "optim_step":
            loss = (parameter * torch.randn(3)).square().sum()
            if plan != "forward":
                loss.backward()
            result = loss.detach().clone()
        else:
            result = None
        if plan is None or plan == "optim_step":
            optimizer.step()
            optimizer.zero_grad()
        return SimpleNamespace(result=lambda: result)

    executor.run = execute
    budget = BufferBudget(64)
    results = []
    execution = WorkerExecution(
        executor,
        SimpleNamespace(scope=SCOPE, primary="step"),
        budget,
        lambda call, value, memory: results.append(value),
        randomness=streams,
    )
    return executor, streams, budget, execution, results


def invoke(execution, operation, *, masked=False, plan=True):
    execution(
        SimpleNamespace(
            command=SimpleNamespace(
                scope=SCOPE,
                skipped=masked,
                accumulate_gradients=operation == "forward_backward",
                request=SimpleNamespace(operation=operation, runtime_trace=False),
            ),
            plan=("forward" if masked else operation) if plan else None,
            hyperparams={},
            input_loader=lambda budget: None,
        )
    )


def save(tmp_path, executor, streams, budget):
    plans = ArtifactRef("plans", "b" * 64, 1)
    info = CheckpointInfo(
        "session",
        "run",
        "history",
        "save-1",
        1,
        1,
        plans,
        plans,
        sys.byteorder,
        metadata=save_call().request.save.metadata,
    )
    wire = io.BytesIO()
    writer = WorkerCheckpointWriter(
        executor, budget, wire, plans=plans, chunk_bytes=24, randomness=streams
    )
    command = WorkerCall(
        SCOPE, 1, save_call().request, None, False, False, checkpoint=info
    )
    call = writer.prepare(BoundWorkerCall(command, None, CallControls(), {}, ()))
    saved = writer(call)
    wire.seek(0)
    path = tmp_path / "rank.ssp"
    with path.open("wb") as destination:
        receive_stream(wire, destination, size_bytes=call.checkpoint.size_bytes)
    return path, RankRestore(
        ArtifactRef("generation", "c" * 64, 1, CHECKPOINT_MEDIA_TYPE),
        info,
        call.checkpoint.reference("rank", saved.sha256),
    )


@pytest.mark.parametrize("split", [False, True], ids=["step", "forward_backward"])
def test_worker_checkpoint_replays_next_update_and_independent_evaluation(
    tmp_path, split
):
    executor, streams, budget, execution, results = runtime()
    invoke(execution, "step")
    invoke(execution, "forward")
    path, source = save(tmp_path, executor, streams, budget)

    def update(interpose):
        if split:
            invoke(execution, "forward_backward")
            if interpose:
                invoke(execution, "forward")
                invoke(execution, "step", masked=True)
            invoke(execution, "forward_backward")
            invoke(execution, "optim_step")
        else:
            if interpose:
                invoke(execution, "forward")
                invoke(execution, "step", masked=True)
            invoke(execution, "step")
        return executor.values["parameter"].clone(), draw()

    expected, expected_draw = update(True)
    expected_evaluation = (
        [value.clone() for value in results[-5:-2]] if split else results[-3:-1]
    )
    reader = WorkerCheckpointReader(
        executor,
        budget,
        lambda _: path,
        chunk_bytes=24,
        world_size=1,
        randomness=streams,
    )
    reader.load(source, reader.prepare_source(source))
    results.clear()
    replay, replay_draw = update(True)
    torch.testing.assert_close(replay, expected, rtol=0, atol=0)
    assert replay_draw == expected_draw and streams.root_seed == 57
    actual_evaluation = results[:3] if split else results[:2]
    for actual, expected_value in zip(
        actual_evaluation, expected_evaluation, strict=True
    ):
        torch.testing.assert_close(actual, expected_value, rtol=0, atol=0)
    reader.load(source, reader.prepare_source(source))
    without_evaluation, without_draw = update(False)
    torch.testing.assert_close(without_evaluation, expected, rtol=0, atol=0)
    assert without_draw == expected_draw and budget.used == (0, 0)


@pytest.mark.parametrize("fault", ["missing", "payload", "rank", "representation"])
def test_bad_rng_preflight_never_mutates_weights_or_default_streams(tmp_path, fault):
    executor, streams, budget, _, _ = runtime()
    path, source = save(tmp_path, executor, streams, budget)
    saved = streams.capture()
    metadata = source.info.rank_metadata()
    if fault != "missing":
        metadata["randomness"] = streams.capture()
        row = metadata["randomness"]["streams"][-1]
        if fault == "payload":
            row["payload"] = "null"
        else:
            row["stream"][fault] = 99 if fault == "rank" else "unknown@1"
    save_checkpoint(executor, path, metadata=metadata, chunk_bytes=24)
    before = executor.values["parameter"].clone()
    reader = WorkerCheckpointReader(
        executor,
        budget,
        lambda _: path,
        chunk_bytes=24,
        world_size=1,
        randomness=streams,
    )
    with pytest.raises((ValueError, RuntimeError)):
        reader.prepare_source(source)
    torch.testing.assert_close(executor.values["parameter"], before, rtol=0, atol=0)
    assert streams.capture() == saved and budget.used == (0, 0)


def test_checkpoint_release_rechecks_rng_and_empty_calls_do_not_advance_it(tmp_path):
    executor, streams, budget, execution, _ = runtime()
    _path, source = save(tmp_path, executor, streams, budget)
    saved = streams.capture()
    invoke(execution, "optim_step", masked=True, plan=False)
    assert streams.capture() == saved
    writer = WorkerCheckpointWriter(
        executor,
        budget,
        io.BytesIO(),
        plans=source.info.plans,
        chunk_bytes=24,
        randomness=streams,
    )
    command = WorkerCall(
        SCOPE, 1, save_call().request, None, False, False, checkpoint=source.info
    )
    call = writer.prepare(BoundWorkerCall(command, None, CallControls(), {}, ()))
    draw()
    with pytest.raises(ValueError, match="changed after staging"):
        writer(call)
    assert budget.used == (0, 0)
