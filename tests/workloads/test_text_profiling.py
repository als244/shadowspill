"""Nsight capture excludes warmup and covers every requested rank/step."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import workloads.recipes.text.profiling as module


@pytest.mark.parametrize("rank", [0, 3])
@pytest.mark.parametrize("count", [0, 5])
def test_capture_boundaries_and_per_rank_step_ranges(monkeypatch, rank, count):
    events = []
    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(module.dist, "get_rank", lambda: rank)
    monkeypatch.setattr(module.dist, "barrier", lambda: events.append("barrier"))
    monkeypatch.setattr(
        module.torch.cuda.profiler, "start", lambda: events.append("start")
    )
    monkeypatch.setattr(
        module.torch.cuda.profiler, "stop", lambda: events.append("stop")
    )

    @contextmanager
    def nvtx(label):
        events.append(("enter", label))
        try:
            yield
        finally:
            events.append(("exit", label))

    monkeypatch.setattr(module.torch.cuda.nvtx, "range", nvtx)

    def train(self, data, *, hyperparams=None):
        self.step_count += 1
        events.append(("step", self.step_count))
        return SimpleNamespace(step=self.step_count)

    monkeypatch.setattr(module.Trainer, "step", train)
    trainer = object.__new__(module.NsightTrainer)
    trainer.step_count = 0
    trainer.nsys_start_step = 2
    trainer.nsys_steps = count
    for number in range(1, 8):
        assert trainer.step(None).step == number

    expected = [("step", 1)]
    if count:
        expected += ["barrier", *(["start"] if rank == 0 else []), "barrier"]
        for number in range(2, 7):
            label = f"train/step={number}/rank={rank}"
            expected += [("enter", label), ("step", number), ("exit", label)]
        expected += ["barrier", *(["stop"] if rank == 0 else []), "barrier"]
    else:
        expected += [("step", number) for number in range(2, 7)]
    expected.append(("step", 7))
    assert events == expected
