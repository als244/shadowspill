"""Durable owner/session/run registration and creation lifecycle, without workers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from threading import RLock
from typing import Self

from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.requests import ArtifactRef, identifier
from shadowspill_client.protocol.runs import CREATION_STAGES, CreationStage, RunInfo
from shadowspill_client.protocol.service import ServiceError

from .artifact_index import reference_bytes
from .journal import Journal
from .journal_format import JournalLimits, JournalRecord
from .requests import RunIdentity
from .run_index import RUN_BYTES, RunIndex, registration_bytes, run_bytes

INDEX_BYTES = 4 << 20


def _identity(info: RunInfo) -> RunIdentity:
    return RunIdentity(info.owner_id, info.session_id, info.run_id)


def _registration(info: RunInfo) -> tuple[object, ...]:
    return (
        _identity(info),
        info.request_sha256,
        info.configuration,
        info.name,
        info.labels,
    )


def _transition(before: RunInfo, after: RunInfo) -> None:
    if (
        _registration(before) != _registration(after)
        or before.server_incarnation != after.server_incarnation
    ):
        raise ValueError("run registration and creation identity are immutable")
    if before.state in ("failed", "stopped", "closed"):
        raise ValueError("terminal run lifecycle records are immutable")
    if before.state == "ready":
        if after.state not in ("stopped", "closed") or (
            after.plans != before.plans
            or after.initial_history_id != before.initial_history_id
            or after.parent_checkpoint != before.parent_checkpoint
            or after.initial_completed_updates != before.initial_completed_updates
        ):
            raise ValueError(
                "ready runs can only stop or close with the original provenance"
            )
    elif after.state == "ready":
        if before.stage != "initializing":
            raise ValueError("worker readiness must follow explicit initialization")
    elif after.state == "preparing":
        assert before.stage is not None and after.stage is not None
        if CREATION_STAGES.index(after.stage) <= CREATION_STAGES.index(before.stage):
            raise ValueError("creation stages cannot restart or move backward")
    elif after.state not in ("failed", "stopped"):
        raise ValueError("unfinished creation cannot close as an initialized run")


class RunCatalog:
    """One service namespace, an authoritative journal and a disposable index.

    A registration acknowledges a creation ID, not GPU readiness. The service
    separately owns bounded creation jobs and live controllers. Restarts expose
    old unfinished runs for request-journal classification before recording stop;
    this catalog never launches, resumes or replays work.
    """

    def __init__(
        self,
        directory: Path,
        *,
        incarnation: str,
        cache_root: Path,
        memory: BufferBudget,
    ) -> None:
        self.incarnation = identifier(incarnation, "server incarnation")
        self._mutex = RLock()
        self.closed = self._broken = self._closing = False
        self._fixed = memory.reserve(INDEX_BYTES, timeout=0)
        try:
            self._journal = Journal(
                directory,
                limits=JournalLimits(
                    record_bytes=RUN_BYTES + 1024,
                    commit_bytes=2 * RUN_BYTES,
                ),
            )
            self._index = RunIndex(cache_root)
            rows = self._journal.records()
            first = next(rows, None)
            if first is None:
                self._journal.append([("catalog.open", {"version": 1})])
            elif (
                first.kind != "catalog.open"
                or dict(first.data) != {"version": 1}
                or type(first.data.get("version")) is not int
            ):
                raise ValueError("unsupported run catalog")
            for row in rows:
                self._apply(row)
        except BaseException:
            if hasattr(self, "_index"):
                self._index.close()
            if hasattr(self, "_journal"):
                self._journal.close()
            self._fixed.close()
            raise

    def _usable(self) -> None:
        if self.closed or self._closing:
            raise RuntimeError("run catalog is closing or closed")
        if self._broken:
            raise RuntimeError("run catalog failed; close and reconcile before reuse")

    def _apply(self, row: JournalRecord) -> None:
        info = RunInfo.from_dict(row.data)
        run_bytes(info)
        before = self._index.get(_identity(info))
        if row.kind == "run.create":
            if (
                before is not None
                or info.state != "preparing"
                or info.stage != "queued"
            ):
                raise ValueError("run creation requires one fresh queued identity")
            self._index.insert(row.position, info)
        elif row.kind == "run.status" and before is not None:
            _transition(before, info)
            self._index.update(info)
        else:
            raise ValueError("unknown or unregistered run lifecycle event")

    def _commit(self, kind: str, info: RunInfo) -> RunInfo:
        run_bytes(info)
        try:
            row = self._journal.append([(kind, info.to_dict())])[0]
            self._apply(row)
        except BaseException:
            self._broken = True
            raise
        return info

    def get(self, identity: RunIdentity) -> RunInfo | None:
        with self._mutex:
            self._usable()
            return self._index.get(identity)

    def create(
        self,
        identity: RunIdentity,
        *,
        request_sha256: str,
        configuration: ArtifactRef,
        name: str | None = None,
        labels: Mapping[str, str] | None = None,
    ) -> tuple[RunInfo, bool]:
        """Commit once before enqueueing; exact retries return the original record."""
        requested = RunInfo(
            identity.owner_id,
            identity.session_id,
            identity.run_id,
            self.incarnation,
            request_sha256,
            configuration,
            name=name,
            labels={} if labels is None else labels,
        )
        registration_bytes(requested)
        with self._mutex:
            self._usable()
            before = self._index.get(identity)
            if before is not None:
                if _registration(before) != _registration(requested):
                    raise ValueError(
                        "run ID already names a different creation request"
                    )
                return before, False
            return self._commit("run.create", requested), True

    def _require(self, identity: RunIdentity, *, live: bool) -> RunInfo:
        self._usable()
        before = self._index.get(identity)
        if before is None:
            raise KeyError("unknown owner/session/run")
        if live and before.server_incarnation != self.incarnation:
            raise RuntimeError("run belongs to an earlier server; classify its loss")
        return before

    def _update(self, before: RunInfo, after: RunInfo) -> RunInfo:
        if before == after:
            return before
        _transition(before, after)
        return self._commit("run.status", after)

    def progress(self, identity: RunIdentity, stage: CreationStage) -> RunInfo:
        with self._mutex:
            before = self._require(identity, live=True)
            return self._update(before, replace(before, stage=stage))

    def ready(
        self,
        identity: RunIdentity,
        *,
        plans: ArtifactRef,
        initial_history_id: str,
        parent_checkpoint: ArtifactRef | None = None,
        initial_completed_updates: int = 0,
    ) -> RunInfo:
        """The caller must have confirmed all workers' actual initialized readiness."""
        reference_bytes(plans)
        if parent_checkpoint is not None:
            reference_bytes(parent_checkpoint)
        with self._mutex:
            before = self._require(identity, live=True)
            return self._update(
                before,
                replace(
                    before,
                    state="ready",
                    stage=None,
                    plans=plans,
                    initial_history_id=initial_history_id,
                    parent_checkpoint=parent_checkpoint,
                    initial_completed_updates=initial_completed_updates,
                ),
            )

    def fail(self, identity: RunIdentity, error: ServiceError) -> RunInfo:
        with self._mutex:
            before = self._require(identity, live=True)
            return self._update(
                before, replace(before, state="failed", stage=None, error=error)
            )

    def stop(self, identity: RunIdentity, error: ServiceError) -> RunInfo:
        with self._mutex:
            before = self._require(identity, live=False)
            return self._update(
                before, replace(before, state="stopped", stage=None, error=error)
            )

    def finish(self, identity: RunIdentity) -> RunInfo:
        """Record explicit close only after the controller confirms worker cleanup."""
        with self._mutex:
            before = self._require(identity, live=True)
            return self._update(before, replace(before, state="closed", stage=None))

    @staticmethod
    def _limit(limit: int) -> None:
        if type(limit) is not int or not 1 <= limit <= 128:
            raise ValueError("run page limit must be in 1..128")

    def page(
        self, owner: str, session: str, *, after: str = "", limit: int = 100
    ) -> tuple[RunInfo, ...]:
        identifier(owner, "owner ID")
        identifier(session, "session ID")
        if after:
            identifier(after, "run cursor")
        self._limit(limit)
        with self._mutex:
            self._usable()
            return self._index.page(owner, session, after, limit)

    def interrupted(
        self, *, after: int = 0, limit: int = 100
    ) -> tuple[tuple[int, RunInfo], ...]:
        if type(after) is not int or after < 0:
            raise ValueError("run recovery cursor must be nonnegative")
        self._limit(limit)
        with self._mutex:
            self._usable()
            return self._index.interrupted(self.incarnation, after, limit)

    def close(self) -> None:
        with self._mutex:
            if not self.closed:
                self._closing = True
                self._index.close()
                self._journal.close()
                self._fixed.close()
                self.closed = True

    def __enter__(self) -> Self:
        self._usable()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
