"""Explicit runtime HTML downloads without engine or plotting dependencies."""

import json
import struct
from pathlib import Path
from tempfile import TemporaryFile
from typing import BinaryIO
from zipfile import ZIP_STORED, ZipFile

from ._async import CallFuture
from ._records import integer
from .artifacts import ArtifactHandle

_CHUNK_BYTES = 128 << 10
_DIRECTORY_BYTES = 64 << 10


def _check_directory(stream: BinaryIO) -> None:
    """Bound ZipFile's metadata allocation for the service's comment-free ZIPs."""
    stream.seek(-22, 2)
    signature, disk, directory_disk, local, total, size, offset, comment = (
        struct.unpack("<4s4H2LH", stream.read(22))
    )
    if (
        signature != b"PK\x05\x06"
        or disk
        or directory_disk
        or comment
        or local != total
    ):
        raise ValueError("invalid runtime trace archive directory")
    if size == 0xFFFFFFFF or offset == 0xFFFFFFFF or total == 0xFFFF:
        stream.seek(-42, 2)
        signature, disk, position, disks = struct.unpack("<4sLQL", stream.read(20))
        if signature != b"PK\x06\x07" or disk or disks != 1:
            raise ValueError("invalid runtime trace ZIP64 locator")
        stream.seek(position)
        fields = struct.unpack("<4sQ2H2L4Q", stream.read(56))
        if fields[:2] != (b"PK\x06\x06", 44) or fields[4:6] != (0, 0):
            raise ValueError("invalid runtime trace ZIP64 directory")
        local, total, size, offset = fields[6:]
    if local != total or total > 8 or size > _DIRECTORY_BYTES:
        raise ValueError("runtime trace archive exceeds its metadata bounds")
    stream.seek(0)


def _copy_report(stream: BinaryIO, destination: Path, max_bytes: int) -> Path:
    _check_directory(stream)
    with ZipFile(stream) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("runtime trace archive contains duplicate members")
        if "report.html" not in names:
            message = "runtime trace contains no HTML report"
            if "report_issue.json" in names:
                issue = archive.getinfo("report_issue.json")
                if issue.file_size <= 8192 and issue.compress_type == ZIP_STORED:
                    message += ": " + str(json.loads(archive.read(issue))["message"])
            raise ValueError(message)
        member = archive.getinfo("report.html")
        if member.file_size > max_bytes or member.compress_type != ZIP_STORED:
            raise ValueError("runtime report exceeds its download bounds")
        with archive.open(member) as source, destination.open("xb") as target:
            try:
                copied = 0
                while chunk := source.read(_CHUNK_BYTES):
                    copied += len(chunk)
                    if copied > max_bytes:
                        raise ValueError("runtime report exceeds its download bounds")
                    target.write(chunk)
            except BaseException:
                destination.unlink()
                raise
    return destination


class TraceHandle(ArtifactHandle):
    """One worker's runtime trace of a compute call.

    A call made with ``runtime_trace=True`` lists one per participating
    worker in `TrainingResult.traces`. The trace is a ZIP archive: as an
    `ArtifactHandle`, its `ArtifactHandle.download` saves the raw archive,
    and `download_report` saves only its HTML report.

    Attributes
    ----------
    reference : ArtifactRef
        The trace archive.
    """

    def download_report(
        self, path: str | Path, *, max_bytes: int = 64 << 20
    ) -> CallFuture[Path]:
        """Download the trace's standalone HTML report into a new local file.

        The report compares the measured execution with the Plan's
        prediction. The archive is downloaded and verified into a temporary
        local file, then only its ``report.html`` is copied to ``path``.

        Parameters
        ----------
        path : str or pathlib.Path
            The local file to create.
        max_bytes : int, default 64 << 20
            Largest report to copy, in bytes (64 MiB by default).

        Returns
        -------
        CallFuture of pathlib.Path
            ``path``, once the file is complete. It fails with `ValueError` if
            the archive has no report, giving the service's reason when it
            recorded one, or the report exceeds ``max_bytes``, and with
            `FileExistsError` if ``path`` exists.

        Raises
        ------
        ValueError
            If ``max_bytes`` is below one, or the handle is not a trace
            archive.
        """
        integer(max_bytes, "runtime report bytes", minimum=1)
        if self.reference.media_type != "application/zip":
            raise ValueError("runtime report requires a trace archive")
        destination = Path(path)

        def transfer() -> Path:
            with TemporaryFile() as source:
                self._service._transport.download(
                    "/v1/artifacts/" + self.id, self.reference, source
                )
                with self._service._memory.reserve(2 * _CHUNK_BYTES + _DIRECTORY_BYTES):
                    return _copy_report(source, destination, max_bytes)

        return self._service._work.submit(transfer)
