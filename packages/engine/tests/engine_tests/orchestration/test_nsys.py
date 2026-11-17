"""Nsight capture excludes warmup and covers every requested rank/step."""

import pytest

from shadowspill.orchestration.capture.nsys import NsysCapture


def test_capture_launch_scopes_metrics_to_the_rank_device(tmp_path):
    capture = NsysCapture(executable="/bin/true", start_step=5)
    command = capture.command(["python", "worker.py"], output=tmp_path / "trace")
    assert command[-2:] == ["python", "worker.py"]
    assert "--capture-range=cudaProfilerApi" in command
    # `stop` and `repeat:1` finalize before the worker releases its host pool.
    assert "--capture-range-end=repeat" in command
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
