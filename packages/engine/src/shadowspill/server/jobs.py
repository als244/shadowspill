"""Bound creation threads and live runs; retire owners independently of clients."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Event, RLock, Thread
from typing import Protocol

from shadowspill_client._records import integer, number
from shadowspill_client.buffers import BufferBudget, BufferLease
from shadowspill_client.protocol.progress import LogChunk, ProgressRequest
from shadowspill_client.protocol.requests import ArtifactRef
from shadowspill_client.protocol.runs import RunInfo, RunKind
from shadowspill_client.protocol.service import ServiceError

from .http import RunAccess
from .live import LiveRun
from .records.requests import RunIdentity
from .records.run_index import registration_bytes
from .storage import ServiceStorage


class Creation(Protocol):
    """One configured installed job, never executable content from a request.

    Construction is resource-free and failure-atomic. ``start`` owns preparation
    then initialization, assigning ``run`` before starting its live resources.
    ``interrupt`` stops only this job's waits/children and preserves all files and
    decoded owners. ``close`` joins its dependencies and releases/clears retained
    objects; the registry calls it only after start and HTTP borrows have ended.
    A failed method retains its resources for retry. No method replays start.
    """

    @property
    def run(self) -> LiveRun | None: ...

    def start(self) -> None: ...

    def interrupt(self) -> None: ...

    def close(self) -> None: ...

    def logs(self, request: ProgressRequest) -> tuple[LogChunk, ...]: ...


@dataclass
class _Job:
    info: RunInfo
    memory: BufferLease
    owner: Creation | None = None
    thread: Thread | None = None
    constructing: bool = True
    live: bool = False
    retiring: bool = False
    users: int = 0
    error: str | None = None
    cleanup_error: str | None = None


def _diagnostic(error: BaseException) -> str:
    return (str(error).strip() or type(error).__name__)[:1024].rstrip()


class RunJobs:
    """One bounded creation/live-owner table and one service maintenance thread.

    Count and the deployment's complete per-job host envelope are reserved before
    registration, factory construction or thread launch. The envelope includes
    decoded families and the creation thread; node-backed I/O arenas and worker
    process claims remain separately admitted. The service's fixed host allowance
    covers this registry and its maintenance thread. Historical runs live only in
    storage. Exact creation retries consult that history without taking a new slot.

    No default idle eviction and no unbounded executor queue. Disconnects do not stop
    preparation or training. Finished/failed owners are reaped without polling
    from a client, while failed cleanup remains counted and retryable. Stop/close
    is allocation lifetime, never a training-request cancellation route. A
    configured idle policy belongs to the sole request controller.
    """

    def __init__(
        self,
        storage: ServiceStorage,
        factory: Callable[[RunInfo], Creation],
        *,
        memory: BufferBudget,
        host_bytes: int,
        max_runs: int,
        reap_interval: float = 0.25,
    ) -> None:
        self.host_bytes = integer(host_bytes, "creation host envelope", minimum=1)
        self.max_runs = integer(max_runs, "maximum owned runs", minimum=1)
        self._interval = number(reap_interval, "retirement interval", minimum=0.01)
        if self._interval > 10 or host_bytes > memory.max_bytes:
            raise ValueError("job host envelope or retirement interval cannot fit")
        self.storage, self._factory, self._memory = storage, factory, memory
        self._jobs: dict[RunIdentity, _Job] = {}
        self._mutex, self._wake = RLock(), Event()
        self._thread: Thread | None = None
        self._closing = self.closed = False

    @property
    def used(self) -> tuple[int, int]:
        with self._mutex:
            return len(self._jobs), sum(job.users for job in self._jobs.values())

    def failures(self) -> tuple[tuple[RunIdentity, str], ...]:
        """Bounded cleanup diagnostics; they do not replace durable run outcomes."""
        with self._mutex:
            return tuple(
                (identity, job.cleanup_error)
                for identity, job in self._jobs.items()
                if job.cleanup_error is not None
            )

    def available(self, identity: RunIdentity) -> bool:
        """A creation reply must wait for the creator's actual thread exit."""
        with self._mutex:
            job = self._jobs.get(identity)
            return (
                not self._closing
                and job is not None
                and job.live
                and not job.retiring
                and job.owner is not None
                and job.owner.run is not None
                and not job.owner.run.finished
            )

    def retired(self, identity: RunIdentity) -> bool:
        """All run owners and the creation envelope have been released."""
        with self._mutex:
            return identity not in self._jobs

    def logs(
        self, identity: RunIdentity, request: ProgressRequest
    ) -> tuple[LogChunk, ...]:
        """Borrow a creation owner without launching or attaching training workers."""
        with self._mutex:
            job = self._jobs.get(identity)
            if job is None or job.owner is None or job.retiring:
                return ()
            job.users += 1
            owner = job.owner
        try:
            return owner.logs(request)
        finally:
            with self._mutex:
                job.users -= 1
            self._wake.set()

    def start(self) -> None:
        with self._mutex:
            if self._thread is not None or self._closing:
                raise RuntimeError("run registry starts once and cannot reopen")
            if not self.storage.recovered:
                raise RuntimeError("recover storage before creating live runs")
            self._thread = Thread(
                target=self._maintain, name="shadowspill-runs", daemon=True
            )
            try:
                self._thread.start()
            except BaseException:
                self._closing = True
                raise

    def create(
        self,
        identity: RunIdentity,
        *,
        request_sha256: str,
        configuration: ArtifactRef,
        name: str | None = None,
        labels: Mapping[str, str] | None = None,
        kind: RunKind = "training",
        publish: Callable[[], None] | None = None,
    ) -> RunInfo:
        """Publish optional creation metadata after admission, before registration.

        Publication runs outside the registry lock and is never repeated for a
        registered identity. A publishing identity occupies its slot/host envelope
        but is not yet acknowledged; concurrent attempts receive backpressure.
        The callback must finish its own I/O before returning or raising.
        """
        with self._mutex:
            if self._thread is None or self._closing:
                raise RuntimeError("run registry is not accepting creations")
            catalog = self.storage.catalog
            if catalog.get(identity) is not None:
                # Reuse the catalog's exact immutable-request comparison even at
                # capacity or after worker retirement. Never reconstruct a job.
                return catalog.create(
                    identity,
                    request_sha256=request_sha256,
                    configuration=configuration,
                    name=name,
                    labels=labels,
                    kind=kind,
                )[0]
            if identity in self._jobs:
                raise TimeoutError("creation metadata is already being published")
            if len(self._jobs) >= self.max_runs:
                raise TimeoutError("all creation/live-run slots are occupied")
            info = RunInfo(
                identity.owner_id,
                identity.session_id,
                identity.run_id,
                self.storage.incarnation,
                request_sha256,
                configuration,
                name=name,
                labels={} if labels is None else labels,
                kind=kind,
            )
            registration_bytes(info)
            memory = self._memory.reserve(self.host_bytes, timeout=0)
            job = _Job(info, memory)
            self._jobs[identity] = job
        try:
            if publish is not None:
                publish()
            with self._mutex:
                job.info, created = catalog.create(
                    identity,
                    request_sha256=request_sha256,
                    configuration=configuration,
                    name=name,
                    labels=labels,
                    kind=kind,
                )
                if not created:
                    del self._jobs[identity]
                    memory.close()
                    self._wake.set()
                    return job.info
        except BaseException:
            # No factory or creation thread exists yet. Even an uncertain catalog
            # commit cannot authorize replay: its poisoned writer requires recovery.
            with self._mutex:
                del self._jobs[identity]
                memory.close()
            self._wake.set()
            raise
        try:
            job.owner = self._factory(info)
            with self._mutex:
                if self._closing:
                    job.error = "Allocation stopped before creation started."
                else:
                    job.thread = Thread(
                        target=self._create,
                        args=(job,),
                        name="shadowspill-create",
                        daemon=True,
                    )
                    try:
                        job.thread.start()
                    except BaseException:
                        if job.thread.ident is None:
                            job.thread = None
                        raise
        except BaseException as error:
            with self._mutex:
                job.error = _diagnostic(error)
        finally:
            with self._mutex:
                job.constructing = False
            self._wake.set()
        return info

    def _create(self, job: _Job) -> None:
        assert job.owner is not None
        try:
            job.owner.start()
        except BaseException as error:
            with self._mutex:
                job.error = _diagnostic(error)
        finally:
            self._wake.set()

    @contextmanager
    def access(self, identity: RunIdentity) -> Iterator[RunAccess]:
        """Keep a live controller reachable through an entire HTTP exchange."""
        with self._mutex:
            job = self._jobs.get(identity)
            if self._closing or job is None or not job.live or job.retiring:
                job, controller = None, None
            else:
                assert job.owner is not None and job.owner.run is not None
                controller = job.owner.run.controller
                job.users += 1
        try:
            with self.storage.requests(identity) as records:
                yield RunAccess(records, controller)
        finally:
            # Do not retain the controller beyond the last counted borrow.
            controller = None
            if job is not None:
                with self._mutex:
                    job.users -= 1
                    assert job.users >= 0
                self._wake.set()

    def _finish(self, identity: RunIdentity, job: _Job) -> bool:
        with self._mutex:
            if job.constructing:
                return False
            running = job.thread is not None and job.thread.is_alive()
            if running and not self._closing:
                return False
            if not running and job.thread is not None:
                # Join the actual thread, not just an event preceding its return.
                job.thread.join()
                job.thread = None
            owner = job.owner
            run = None if owner is None else owner.run
            if not running and job.error is None and not self._closing:
                info = self.storage.catalog.get(identity)
                if info is not None and info.state in ("prepared", "planned"):
                    pass  # Completed preparation has no persistent execution run.
                elif run is None or not run.ready:
                    job.error = "Creation returned without initialized workers."
                elif not run.finished:
                    job.live = True
                    return False
            job.retiring = True
            interrupt = (
                self._closing
                or job.error is not None
                or (
                    run is not None
                    and not run.closed
                    and run.controller.failure is not None
                )
            )
        if owner is not None and interrupt:
            owner.interrupt()
        if running:
            return False
        with self._mutex:
            if job.users:
                return False
        if owner is not None:
            owner.close()
        info = self.storage.catalog.get(identity)
        if info is not None and info.state == "preparing":
            if self._closing:
                self.storage.catalog.stop(
                    identity,
                    ServiceError("allocation_stopped", "Allocation ownership ended."),
                )
            else:
                self.storage.catalog.fail(
                    identity,
                    ServiceError("creation_failed", job.error or "Creation failed."),
                )
        elif info is None or info.state == "ready":
            raise RuntimeError("retired creation did not record its terminal run state")
        return True

    def _maintain(self) -> None:
        while True:
            self._wake.wait(self._interval)
            self._wake.clear()
            with self._mutex:
                jobs = tuple(self._jobs.items())
            for identity, job in jobs:
                try:
                    retired = self._finish(identity, job)
                except BaseException as error:
                    with self._mutex:
                        job.cleanup_error = _diagnostic(error)
                    continue
                if retired:
                    # _finish's local references have returned. Clear remaining
                    # job/thread ownership before releasing its full host envelope.
                    with self._mutex:
                        job.owner = None
                        del self._jobs[identity]
                        job.memory.close()
                else:
                    with self._mutex:
                        job.cleanup_error = None
            with self._mutex:
                if self._closing and not self._jobs:
                    self.closed = True
                    return

    def close(self, *, timeout: float = 10) -> None:
        """Request allocation stop and wait boundedly; failed owners remain live."""
        number(timeout, "registry close timeout", minimum=0)
        with self._mutex:
            self._closing = True
            thread = self._thread
            self._wake.set()
        if thread is not None and thread.ident is not None:
            thread.join(timeout=timeout)
        with self._mutex:
            if self._jobs or (thread is not None and thread.is_alive()):
                raise RuntimeError("run jobs still own creation, borrows or cleanup")
            self.closed = True
            self._thread = None
