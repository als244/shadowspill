"""Retain Nsight collector files until process exit and verified artifact transfer."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
from typing import BinaryIO, Literal

from shadowspill.orchestration.launch.admission import NodeLease
from shadowspill.orchestration.launch.publication import (
    OutputFiles,
    OutputSpec,
    _open_regular,
    require_output,
)
from shadowspill.orchestration.launch.staging import COPY_BYTES
from shadowspill.orchestration.launch.supervision import OwnedProcess
from shadowspill_client._records import integer, number, text
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.captures import (
    NSIGHT_MEDIA_TYPE,
    WINDOW_BYTES,
    NsightWindow,
)
from shadowspill_client.protocol.requests import ArtifactRef

from .nsys import NsysCapture
from .progress import read_progress


@dataclass(frozen=True)
class CollectorLimits:
    """Deployment allowances for the external profiler, as for phase processes.

    Host/work-directory bounds need qualification for the installed Nsight
    version and workload; reservations are not OS quotas. Final file count/size
    and report checksum checks reject excess before publication. Report storage
    is also reserved through the existing OutputFiles owner before launch.
    """

    host_bytes: int = 1 << 30
    disk_bytes: int = 2 << 30
    max_files: int = 1024
    finish_seconds: float = 120
    executable: str = "nsys"

    def __post_init__(self) -> None:
        for name in ("host_bytes", "disk_bytes", "max_files"):
            integer(getattr(self, name), f"collector {name}", minimum=1)
        if (
            not 1
            <= number(self.finish_seconds, "collector finalization seconds")
            <= 3600
        ):
            raise ValueError("collector finalization allowance must be 1-3600 seconds")
        text(self.executable, "installed collector executable")


@dataclass(frozen=True)
class CaptureOutcome:
    window: NsightWindow
    status: Literal["ready", "failed", "missing"]
    report: ArtifactRef | None
    error: str | None


class CaptureFiles:
    """Use the existing output owner; the lease alone proves producers stopped."""

    def __init__(
        self,
        lease: NodeLease,
        settings: NsysCapture,
        limits: CollectorLimits,
        *,
        artifact_id: str,
        root: Path,
        disk: BufferBudget,
        memory: BufferBudget,
        outputs: OutputFiles,
        device_id: str | None = None,
    ) -> None:
        if lease.closed:
            raise ValueError("capture requires a live node lease")
        if device_id is not None and device_id not in lease.request.devices:
            raise ValueError("collector device is not reserved")
        self.settings = replace(settings, executable=limits.executable, output_dir=None)
        self.limits, self._lease = limits, lease
        self._device = device_id
        self._memory = memory.reserve(
            4096 + 4 * WINDOW_BYTES + COPY_BYTES + 512 * limits.max_files, timeout=0
        )
        try:
            self._disk = disk.reserve(limits.disk_bytes, timeout=0)
        except BaseException:
            self._memory.close()
            raise
        try:
            self._directory = TemporaryDirectory(prefix="nsys-", dir=root)
        except BaseException:
            self._disk.close()
            self._memory.close()
            raise
        self.directory = Path(self._directory.name)
        self.progress_path = self.directory / "window.json"
        self._report = self.directory / "trace.nsys-rep"
        try:
            (self.directory / "tmp").mkdir(mode=0o700)
            self._output = outputs.reserve(
                OutputSpec(
                    artifact_id, limits.disk_bytes, NSIGHT_MEDIA_TYPE, exact_size=False
                )
            )
        except BaseException:
            self._directory.cleanup()
            self._disk.close()
            self._memory.close()
            raise
        self._root = outputs.root
        self._process: OwnedProcess | None = None
        self._outcome: CaptureOutcome | None = None
        self._lock = RLock()
        self._finished = self.closed = False
        self._output.begin()

    @property
    def environment(self) -> Mapping[str, str]:
        return {"TMPDIR": str(self.directory / "tmp")}

    def check(self, lease: NodeLease, device_id: str) -> None:
        if self.closed or self._lease is not lease or self._device != device_id:
            raise ValueError("collector belongs to another resource owner or device")

    def command(self, worker: Sequence[str]) -> list[str]:
        if self.closed or self._finished:
            raise RuntimeError("capture ownership is no longer live")
        return self.settings.command(worker, output=self._report.with_suffix(""))

    def launched(self, process: OwnedProcess) -> None:
        if self._process is not None or self.closed or self._finished:
            raise RuntimeError("collector can only launch once")
        self._process = process

    def _check_files(self) -> None:
        pending, count, size = [self.directory], 0, 0
        while pending:
            with os.scandir(pending.pop()) as entries:
                for entry in entries:
                    count += 1
                    if count > self.limits.max_files:
                        raise ValueError("collector file count exceeds its envelope")
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        size += entry.stat(follow_symlinks=False).st_size
                        if size > self.limits.disk_bytes:
                            raise ValueError(
                                "collector files exceed their disk envelope"
                            )
                    else:
                        raise ValueError("collector output contains a nonregular entry")

    def outcome(self) -> CaptureOutcome:
        with self._lock:
            return self._inspect()

    def _inspect(self) -> CaptureOutcome:
        if self.closed or not self._lease.closed:
            raise RuntimeError("collector producers have not confirmed termination")
        if self._outcome is not None:
            return self._outcome
        if not self._finished:
            self._output.finished()
            self._finished = True
        window = NsightWindow(
            self.settings.start_step,
            self.settings.steps,
            "missing",
            error="Worker produced no capture progress.",
        )
        reference = None
        status: Literal["ready", "failed", "missing"]
        error: str | None
        try:
            self._check_files()
            if self.progress_path.exists():
                window = read_progress(self.progress_path)
                if (window.start_step, window.steps) != (
                    self.settings.start_step,
                    self.settings.steps,
                ):
                    raise ValueError(
                        "capture progress differs from the selected window"
                    )
            if self._report.exists():
                with _open_regular(self._report) as stream:
                    size = os.fstat(stream.fileno()).st_size
                    if not 0 < size <= self._output.spec.size_bytes:
                        raise ValueError(
                            "Nsight report is empty or exceeds its envelope"
                        )
                    digest, remaining = hashlib.sha256(), size
                    while remaining:
                        block = stream.read(min(COPY_BYTES, remaining))
                        if not block:
                            raise ValueError("Nsight report became truncated")
                        digest.update(block)
                        remaining -= len(block)
                    if stream.read(1):
                        raise ValueError("Nsight report grew after producer exit")
                reference = ArtifactRef(
                    self._output.spec.id, digest.hexdigest(), size, NSIGHT_MEDIA_TYPE
                )
                destination = require_output(self._root, self._output.spec)
                os.link(self._report, destination, follow_symlinks=False)
                self._output.verify(reference)
            if self._process is None or self._process.returncode != 0:
                status, error = "failed", "Collector did not exit successfully."
            elif reference is None:
                status, error = "missing", "Collector produced no Nsight report."
            elif window.state != "complete" or window.history_id is None:
                status, error = (
                    "failed",
                    "Worker did not confirm the complete selected update window.",
                )
            else:
                status, error = "ready", None
        except (OSError, ValueError) as error_value:
            status, reference = "failed", None
            error = (str(error_value).strip() or type(error_value).__name__)[
                :1024
            ].rstrip()
        self._outcome = CaptureOutcome(window, status, reference, error)
        return self._outcome

    def open(self, reference: ArtifactRef) -> AbstractContextManager[BinaryIO]:
        if self.outcome().report != reference:
            raise ValueError("capture reference differs from finalized content")
        return self._output.open(reference)

    def close(self) -> None:
        with self._lock:
            self._close()

    def _close(self) -> None:
        if self.closed:
            return
        if not self._lease.closed:
            raise RuntimeError("collector producers have not confirmed termination")
        if not self._finished:
            self._output.finished()
            self._finished = True
        self._output.close()
        self._directory.cleanup()
        self._disk.close()
        self._memory.close()
        self.closed = True
