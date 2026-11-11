"""Nsight capture excludes warmup and covers every requested rank/step."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import orchestration._nsys_trainer as module
from orchestration.nsys import NsysCapture, nsight_trainer


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

    class Trainer:
        def __init__(self):
            self.step_count = 0

        def step(self, **inputs):
            annotated = inputs.pop("profiler_annotations", False)
            assert annotated == (count > 0 and 2 <= self.step_count + 1 <= 6)
            assert inputs == {
                "args": [("tokens",)],
                "kwargs": [{"normalizer": 0.5}],
                "hyperparams": {"lr": 0.01},
            }
            self.step_count += 1
            events.append(("step", self.step_count))
            return SimpleNamespace(step=self.step_count)

    factory = nsight_trainer(Trainer)
    assert nsight_trainer(factory) is factory  # Never nest two collectors.
    trainer = factory(nsys_start_step=2, nsys_steps=count)
    for number in range(1, 8):
        assert (
            trainer.step(
                args=[("tokens",)],
                kwargs=[{"normalizer": 0.5}],
                hyperparams={"lr": 0.01},
            ).step
            == number
        )

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


def test_capture_launch_wraps_worker_tree_and_preserves_device_mask(tmp_path):
    capture = NsysCapture(executable="/bin/true", start_step=5)
    command = capture.command(
        ["python", "-m", "torch.distributed.run"], output=tmp_path / "trace"
    )
    assert command[-3:] == ["python", "-m", "torch.distributed.run"]
    assert "--capture-range=cudaProfilerApi" in command
    assert "--capture-range-end=stop" in command
    assert "--wait=all" in command
    assert "--gpu-metrics-devices=cuda-visible" in command
    assert list(capture.step_numbers) == [5, 6]
    without = NsysCapture(executable="/bin/true", gpu_metrics=False).command(
        [], output=tmp_path / "trace"
    )
    assert not any(part.startswith("--gpu-metrics") for part in without)
    with pytest.raises(FileNotFoundError, match="Nsight Systems executable"):
        NsysCapture(executable="/does/not/exist").command([], output=tmp_path / "trace")


@pytest.mark.parametrize(
    "options", [{"steps": 0}, {"start_step": True}, {"gpu_metrics_frequency": 0}]
)
def test_invalid_capture_is_rejected_before_launch(options):
    with pytest.raises(ValueError):
        NsysCapture(**options)
