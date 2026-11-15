"""Native reports share existing output ownership and require actual producer exit."""

import os
import sys
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from shadowspill.orchestration.capture.files import CaptureFiles, CollectorLimits
from shadowspill.orchestration.capture.nsys import NsysCapture
from shadowspill.orchestration.capture.progress import write_progress
from shadowspill.orchestration.launch.admission import (
    NodeAdmission,
    NodeCapacity,
    ResourceRequest,
)
from shadowspill.orchestration.launch.publication import OutputFiles
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.captures import NsightWindow


@contextmanager
def capture(tmp_path, *, limits=None, returncode=0, launch=True):
    node = NodeAdmission(NodeCapacity("local", (), 1 << 20))
    lease = node.reserve(ResourceRequest({}, 1 << 20))
    memory, disk = BufferBudget(2 << 20, 16), BufferBudget(2 << 20, 16)
    limits = limits or CollectorLimits(
        host_bytes=1 << 20, disk_bytes=128 << 10, max_files=16, executable="/bin/true"
    )
    with OutputFiles(root=tmp_path, disk=disk, memory=memory) as outputs:
        files = CaptureFiles(
            lease,
            NsysCapture(),
            limits,
            artifact_id="nsys-report",
            root=tmp_path,
            disk=disk,
            memory=memory,
            outputs=outputs,
        )
        if launch:
            process = lease.launch_command(
                [sys.executable, "-c", f"raise SystemExit({returncode})"],
                devices=(),
                environment={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                directory=tmp_path / "logs",
                cwd=tmp_path,
            )
            files.launched(process)
        try:
            yield SimpleNamespace(
                node=node,
                lease=lease,
                files=files,
                outputs=outputs,
                memory=memory,
                disk=disk,
            )
        finally:
            lease.close()
            files.close()
    assert node.used.claims == 0 and memory.used == disk.used == (0, 0)


def ready(files):
    write_progress(
        files.progress_path,
        NsightWindow(2, 2, "complete", 2, "history", "backward", "optimizer"),
    )
    (files.directory / "trace.nsys-rep").write_bytes(b"sentinel native report")


def test_reports_require_confirmed_exit_and_retain_output_until_transfer_ends(tmp_path):
    with capture(tmp_path) as ctx:
        ready(ctx.files)
        held = ctx.disk.used
        command = ctx.files.command([sys.executable, "worker.py"])
        assert "--capture-range-end=repeat" in command
        assert command[-2:] == [sys.executable, "worker.py"]
        assert ctx.files.environment == {"TMPDIR": str(ctx.files.directory / "tmp")}
        with pytest.raises(RuntimeError, match="termination"):
            ctx.files.outcome()
        with pytest.raises(RuntimeError, match="termination"):
            ctx.files.close()
        ctx.lease.close()
        outcome = ctx.files.outcome()
        assert outcome.status == "ready" and outcome.error is None
        assert outcome.window.completed_updates == 2
        assert ctx.files.outcome() == outcome and ctx.disk.used == held
        with ctx.files.open(outcome.report) as stream:
            with pytest.raises(RuntimeError, match="active producer/consumer"):
                ctx.files.close()
            assert stream.read() == b"sentinel native report"
            assert ctx.disk.used == held
        ctx.files.close()
        assert ctx.disk.used == (0, 0)


@pytest.mark.parametrize(
    "fault",
    [
        "no_report",
        "no_window",
        "partial",
        "wrong_window",
        "bytes",
        "files",
        "symlink",
        "empty",
        "metadata",
    ],
)
def test_invalid_or_incomplete_capture_never_becomes_a_ready_report(tmp_path, fault):
    with capture(tmp_path) as ctx:
        ready(ctx.files)
        report = ctx.files.directory / "trace.nsys-rep"
        if fault == "no_report":
            report.unlink()
        elif fault == "no_window":
            ctx.files.progress_path.unlink()
        elif fault == "partial":
            write_progress(
                ctx.files.progress_path,
                NsightWindow(2, 2, "running", 1, "history", "first", "one"),
            )
        elif fault == "wrong_window":
            write_progress(
                ctx.files.progress_path,
                NsightWindow(3, 2, "complete", 2, "history", "first", "last"),
            )
        elif fault == "bytes":
            report.write_bytes(b"x" * ctx.files.limits.disk_bytes)
        elif fault == "files":
            for index in range(ctx.files.limits.max_files):
                (ctx.files.directory / f"extra-{index}").touch()
        elif fault == "symlink":
            report.unlink()
            report.symlink_to(ctx.files.progress_path)
        elif fault == "empty":
            report.write_bytes(b"")
        else:
            ctx.files.progress_path.write_bytes(b"malformed")
        ctx.lease.close()
        result = ctx.files.outcome()
        assert result.status == ("missing" if fault == "no_report" else "failed")
        assert result.error
        assert (result.report is not None) == (fault in ("no_window", "partial"))


@pytest.mark.parametrize("launch,returncode", [(False, 0), (True, 17)])
def test_collector_exit_failure_survives_complete_worker_evidence(
    tmp_path, launch, returncode
):
    with capture(tmp_path, launch=launch, returncode=returncode) as ctx:
        ready(ctx.files)
        ctx.lease.close()
        result = ctx.files.outcome()
        assert result.window.state == "complete" and result.status == "failed"
        assert result.report is not None and result.error


def test_failed_reservation_releases_capture_metadata_and_disk(tmp_path):
    node = NodeAdmission(NodeCapacity("local", (), 1 << 20))
    memory, disk = BufferBudget(2 << 20, 16), BufferBudget(128 << 10, 16)
    with (
        node.reserve(ResourceRequest({}, 1 << 20)) as lease,
        OutputFiles(root=tmp_path, memory=memory, disk=disk) as outputs,
    ):
        before = memory.used
        with pytest.raises((ValueError, TimeoutError)):
            CaptureFiles(
                lease,
                NsysCapture(),
                replace(CollectorLimits(), disk_bytes=128 << 10),
                artifact_id="report",
                root=tmp_path,
                memory=memory,
                disk=disk,
                outputs=outputs,
            )
        assert memory.used == before and disk.used == (0, 0)
        assert not tuple(tmp_path.glob("nsys-*"))
    assert memory.used == disk.used == (0, 0)
