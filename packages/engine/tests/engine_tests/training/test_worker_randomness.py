"""Actual default CPU generators; worker ownership does not require a GPU."""

import copy
import json
import random

import numpy as np
import pytest
import torch

from engine_tests.execution.test_checkpoint import MemoryExecutor
from shadowspill.checkpoint import load_checkpoint, save_checkpoint
from shadowspill.pytorch._rng import preserve_rng
from shadowspill.training.worker.randomness import WorkerRandomness


@pytest.fixture(autouse=True)
def preserved():
    with preserve_rng(torch.device("cpu")):
        yield


def draw():
    return [random.random(), *np.random.rand(4), *torch.rand(4).tolist()]


def owner(seed=57, rank=0):
    result = WorkerRandomness(torch.device("cpu"), rank)
    result.initialize(seed)
    return result


def test_seed_replays_training_and_evaluation_keeps_its_own_advancing_stream():
    first = owner()
    checkpoint = first.capture()
    expected = draw()
    first.restore(checkpoint)
    with first.evaluation():
        evaluation_one = draw()
    with first.evaluation():
        evaluation_two = draw()
    assert evaluation_one != evaluation_two and draw() == expected
    second = owner()
    assert draw() == expected
    with second.evaluation():
        assert draw() == evaluation_one
    with second.evaluation():
        assert draw() == evaluation_two
    distinct = owner(rank=1)
    assert draw() != expected
    assert all(stream.rank == 1 for stream in distinct.streams)


def test_failed_evaluation_preserves_training_and_snapshot_replays_both_streams():
    streams = owner()
    saved = streams.capture()
    expected = draw()
    streams.restore(saved)
    with pytest.raises(OSError, match="evaluation failed"), streams.evaluation():
        evaluated = draw()
        with pytest.raises(RuntimeError, match="boundary"):
            streams.capture()
        raise OSError("evaluation failed")
    assert draw() == expected
    streams.restore(saved)
    with streams.evaluation():
        assert draw() == evaluated
        with pytest.raises(RuntimeError, match="idle"), streams.evaluation():
            pass


def test_native_checkpoint_roundtrip_replays_training_and_evaluation(tmp_path):
    streams = owner()
    draw()
    with streams.evaluation():
        draw()
    path = tmp_path / "checkpoint.ssp"
    save_checkpoint(
        MemoryExecutor(),
        path,
        metadata={"randomness": streams.capture()},
        chunk_bytes=17,
    )
    expected_training = draw()
    with streams.evaluation():
        expected_evaluation = draw()
    restored = WorkerRandomness(torch.device("cpu"), 0)
    metadata = load_checkpoint(MemoryExecutor(), path, chunk_bytes=19)
    restored.restore(metadata["randomness"])
    assert restored.root_seed == 57 and draw() == expected_training
    with restored.evaluation():
        assert draw() == expected_evaluation


@pytest.mark.parametrize(
    "fault",
    ["version", "rank", "representation", "duplicate", "missing", "payload", "extra"],
)
def test_bad_snapshot_fails_before_any_default_stream_mutates(fault):
    streams = owner()
    saved = streams.capture()
    changed = copy.deepcopy(saved)
    if fault == "version":
        changed["version"] = True
    elif fault == "rank":
        changed["streams"][-1]["stream"]["rank"] = 7
    elif fault == "representation":
        changed["streams"][-1]["stream"]["representation"] = "unknown@1"
    elif fault == "duplicate":
        changed["streams"].append(changed["streams"][0])
    elif fault == "missing":
        changed["streams"].pop()
    elif fault == "payload":
        changed["streams"][-1]["payload"] = json.dumps(["tensor", "uint8", [2], "AAA="])
    else:
        changed["unknown"] = 1
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        streams.restore(changed)
    assert streams.capture() == saved


def test_preparation_is_read_only_and_does_not_use_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("CPU RNG preparation touched CUDA")

    monkeypatch.setattr(torch.cuda, "get_rng_state", forbidden)
    monkeypatch.setattr(torch.cuda, "set_rng_state", forbidden)
    streams = owner()
    saved = streams.capture()
    streams.prepare(saved)
    assert streams.capture() == saved
    with pytest.raises(RuntimeError, match="already initialized"):
        streams.initialize(1)


def test_inert_owner_requires_initialization_or_restore():
    streams = WorkerRandomness(torch.device("cpu"), 0)
    with pytest.raises(RuntimeError, match="boundary"):
        streams.capture()
    with pytest.raises(RuntimeError, match="initialized"), streams.evaluation():
        pass
