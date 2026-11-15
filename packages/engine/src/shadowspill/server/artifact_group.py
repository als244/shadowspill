"""Owned persistent files between completed production and durable publication."""

from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Self

from shadowspill_client.protocol.requests import ArtifactRef

from .records.artifact_index import ArtifactAlias

if TYPE_CHECKING:
    from .artifacts import ArtifactStore, _Publication


class StagedArtifacts:
    """A CompletedWork owner, retained through execution receipt and publication.

    Construction is private to ArtifactStore after every producer has returned.
    No worker may still be writing these files. Closing an uncommitted group
    discards it; a commit error retains uncertain data for journal recovery.
    """

    def __init__(
        self,
        store: ArtifactStore,
        pending: _Publication,
        references: tuple[ArtifactRef, ...],
        paths: tuple[Path, ...],
        alias: ArtifactAlias | None,
    ) -> None:
        self._store, self._pending = store, pending
        self.references, self._paths, self._alias = references, paths, alias
        self._mutex = RLock()
        self.closed = False

    def commit(self) -> tuple[ArtifactRef, ...]:
        with self._mutex:
            if self.closed:
                raise RuntimeError("staged artifacts are closed")
            if self._pending.uncertain:
                raise RuntimeError(
                    "artifact commit is uncertain; reconcile before reuse"
                )
            if not self._pending.committed:
                self._store._commit(
                    self._pending, self.references, self._paths, self._alias
                )
            return self.references

    def close(self) -> None:
        with self._mutex:
            if not self.closed:
                self._store._finish(self._pending)
                self.closed = True

    def __enter__(self) -> Self:
        if self.closed:
            raise RuntimeError("staged artifacts are closed")
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
