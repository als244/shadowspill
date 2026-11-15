"""Private capture progress is bounded metadata, independent of a native report."""

import pytest

from shadowspill.orchestration.capture.nsys import NsysCapture
from shadowspill.orchestration.capture.progress import read_progress, write_progress
from shadowspill_client.protocol.captures import WINDOW_BYTES, NsightWindow
from shadowspill_client.protocol.codec import ProtocolError


def test_progress_replacement_is_bounded_and_rejects_changed_nonregular_input(tmp_path):
    path = tmp_path / "window.json"
    pending = NsightWindow(2, 2)
    write_progress(path, pending)
    assert read_progress(path) == pending and path.stat().st_size < WINDOW_BYTES
    complete = NsightWindow(2, 2, "complete", 2, "history", "first", "last")
    write_progress(path, complete)
    assert read_progress(path) == complete and tuple(tmp_path.iterdir()) == (path,)
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(OSError):
        read_progress(link)
    path.write_bytes(b"x" * (WINDOW_BYTES + 1))
    with pytest.raises(ProtocolError, match="exceeds"):
        read_progress(path)


def test_service_capture_settings_preserve_defaults_and_reject_remote_paths():
    assert NsysCapture.from_settings(None) is None
    assert NsysCapture.from_settings({}) == NsysCapture()
    assert NsysCapture.from_settings({"steps": 3, "gpu_metrics": False}) == NsysCapture(
        steps=3, gpu_metrics=False
    )
    for settings in (
        {"executable": "arbitrary-command"},
        {"output_dir": "/unowned/path"},
        {"start_step": 0},
        {"steps": [1]},
        {"gpu_metrics": 1},
        {"gpu_metrics_frequency": 200001},
    ):
        with pytest.raises((TypeError, ValueError)):
            NsysCapture.from_settings(settings)
