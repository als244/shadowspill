"""Laptop runtime reports copy only the verified bounded HTML member."""

import io
import json
import struct
from zipfile import ZIP_STORED, ZipFile

import pytest

from shadowspill_client.traces import _copy_report


def archive(*, report=b"<html>measured</html>", issue=None):
    stream = io.BytesIO()
    with ZipFile(stream, "w", compression=ZIP_STORED) as output:
        output.writestr("step.json", b"{}")
        if report is not None:
            output.writestr("report.html", report)
        if issue is not None:
            output.writestr("report_issue.json", json.dumps({"message": issue}))
    return stream


def test_runtime_report_is_explicit_and_never_overwrites(tmp_path):
    path = tmp_path / "runtime.html"
    assert _copy_report(archive(), path, 1024) == path
    assert path.read_bytes() == b"<html>measured</html>"
    with pytest.raises(FileExistsError):
        _copy_report(archive(), path, 1024)
    assert path.read_bytes() == b"<html>measured</html>"


def test_runtime_report_bounds_and_failures_keep_raw_evidence(tmp_path):
    path = tmp_path / "runtime.html"
    with pytest.raises(ValueError, match="download bounds"):
        _copy_report(archive(report=b"a" * 2048), path, 1024)
    with pytest.raises(ValueError, match="render failed"):
        _copy_report(archive(report=None, issue="render failed"), path, 1024)
    with pytest.raises(ValueError, match="no HTML"):
        _copy_report(archive(report=None), path, 1024)
    damaged = bytearray(archive().getvalue())
    struct.pack_into("<L", damaged, len(damaged) - 10, 1 << 30)
    with pytest.raises(ValueError, match="metadata bounds"):
        _copy_report(io.BytesIO(damaged), path, 1024)
    assert not path.exists()
