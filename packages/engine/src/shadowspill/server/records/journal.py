"""One exclusive journal writer, with atomic durable groups and bounded recovery.

A commit file is visible only after its complete bytes have been fsynced. Linking
it into the committed sequence and fsyncing the directory completes the commit.
Unpublished temporary files are an uncommitted tail; corrupt or missing committed
files are errors. This is a POSIX filesystem backend, not an NFS SQLite WAL.
"""

from __future__ import annotations

import fcntl
import os
import re
import tempfile
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from .journal_format import (
    JournalLimits,
    JournalRecord,
    decode_commit,
    encode_commit,
)

_COMMIT = re.compile(r"[0-9]{20}\.ssj")
_TEMP = ".ssj.pending-"


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class Journal:
    """A storage primitive; request semantics and GPU state remain in the engine.

    The parent directory must already exist beneath configured service storage.
    Limits bound individual reads and commit groups, not retained history size.
    Scan records incrementally; no in-memory copy of the entire journal is kept.
    """

    def __init__(self, directory: str | Path, *, limits: JournalLimits | None = None):
        self.directory = Path(directory)
        self.limits = limits or JournalLimits()
        self._mutex = threading.RLock()
        self._closed = self._broken = False
        self._index = self._position = 0
        self._previous: str | None = None
        try:
            self.directory.mkdir(mode=0o700)
        except FileExistsError:
            if not self.directory.is_dir():
                raise
        _sync_directory(self.directory.parent)
        self._lock = (self.directory / ".writer.lock").open("a+b")
        try:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException as error:
            self._lock.close()
            if isinstance(error, BlockingIOError):
                raise RuntimeError("journal already has an active writer") from error
            raise
        try:
            self._recover()
        except BaseException:
            self.close()
            raise

    @property
    def position(self) -> int:
        with self._mutex:
            self._usable()
            return self._position

    def _path(self, index: int) -> Path:
        return self.directory / f"{index:020d}.ssj"

    def _read(
        self, index: int, position: int, previous: str | None
    ) -> tuple[tuple[JournalRecord, ...], str]:
        with self._path(index).open("rb") as stream:
            data = stream.read(self.limits.commit_bytes + 1)
        return decode_commit(
            data, index=index, position=position, previous=previous, limits=self.limits
        )

    def _recover(self) -> None:
        while self._path(self._index + 1).exists():
            records, checksum = self._read(
                self._index + 1, self._position + 1, self._previous
            )
            self._index += 1
            self._position = records[-1].position
            self._previous = checksum
        # Do not silently turn a gap/corrupt committed prefix into a shorter run.
        with os.scandir(self.directory) as entries:
            for entry in entries:
                if (
                    _COMMIT.fullmatch(entry.name)
                    and not 1 <= int(entry.name[:20]) <= self._index
                ):
                    raise ValueError("journal committed sequence has a gap")
        with os.scandir(self.directory) as entries:
            for entry in entries:
                if entry.name.startswith(_TEMP) and entry.is_file(
                    follow_symlinks=False
                ):
                    Path(entry.path).unlink()
        # A previous writer may have died after publication but before directory
        # sync. Make the recovered visible prefix durable before exposing it.
        _sync_directory(self.directory)

    def append(
        self, entries: Sequence[tuple[str, Mapping[str, Any]]]
    ) -> tuple[JournalRecord, ...]:
        """Acknowledge only a committed group; any I/O failure poisons the writer.

        On an uncertain error, reopen/reconcile records before any new work.
        Retrying a GPU operation is never a responsibility of this journal.
        """
        with self._mutex:
            self._usable()
            data, records, checksum = encode_commit(
                self._index + 1,
                self._position + 1,
                self._previous,
                entries,
                self.limits,
            )
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=self.directory, prefix=_TEMP, delete=False
                ) as stream:
                    temporary = Path(stream.name)
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                # Exclusive publication cannot replace an existing committed file.
                os.link(temporary, self._path(self._index + 1))
                temporary.unlink()
                temporary = None
                _sync_directory(self.directory)
            except BaseException:
                self._broken = True
                raise
            finally:
                if temporary is not None:
                    # Recovery removes only our unpublished temporary tail.
                    with suppress(OSError):
                        temporary.unlink(missing_ok=True)
            self._index += 1
            self._position = records[-1].position
            self._previous = checksum
            return records

    def records(self) -> Iterator[JournalRecord]:
        """Read a committed prefix snapshot, retaining at most one bounded group."""
        with self._mutex:
            self._usable()
            limit = self._index
        position, previous = 1, None
        for index in range(1, limit + 1):
            rows, previous = self._read(index, position, previous)
            yield from rows
            position = rows[-1].position + 1

    def _usable(self) -> None:
        if self._closed:
            raise RuntimeError("journal is closed")
        if self._broken:
            raise RuntimeError(
                "journal durability failed; close and reconcile before reuse"
            )

    def close(self) -> None:
        with self._mutex:
            if not self._closed:
                self._lock.close()
                self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
