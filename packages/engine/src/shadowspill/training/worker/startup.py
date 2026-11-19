"""Exact all-worker startup barriers over the existing private command streams."""

from collections.abc import Callable, Mapping
from contextlib import ExitStack
from typing import BinaryIO, Literal

from shadowspill_client.protocol.codec import ProtocolError

from .channel import read_message, write_message
from .connection import WorkerConnection
from .group import WorkerGroup
from .protocol import CODEC, WorkerBoot, WorkerStartup


class WorkerStartupConnection:
    """A once-only owner, transferred to a command connection after readiness.

    The allocation's node daemon owns transport authentication, startup deadlines and
    process termination. This endpoint never reconnects or resends a start.
    """

    def __init__(
        self, expected: WorkerStartup, reader: BinaryIO, writer: BinaryIO
    ) -> None:
        CODEC.encode(WorkerBoot(expected, "start"))
        self.expected, self._reader, self._writer = expected, reader, writer
        self._phase = "new"

    def _require(self, phase: str) -> None:
        if self._phase != phase:
            raise RuntimeError("worker startup phase is consumed; never restart it")

    def _receive(self, phase: Literal["prepared", "ready"]) -> None:
        if read_message(self._reader) != WorkerBoot(self.expected, phase):
            raise ProtocolError(
                "worker startup evidence is absent or differs from admission"
            )

    def prepared(self) -> None:
        self._require("new")
        self._phase = "failed"
        self._receive("prepared")
        self._phase = "prepared"

    def release(self) -> None:
        self._require("prepared")
        self._phase = "starting"
        try:
            write_message(self._writer, WorkerBoot(self.expected, "start"))
        except BaseException:
            self._phase = "failed"
            raise

    def ready(self) -> WorkerConnection:
        self._require("starting")
        self._phase = "failed"
        self._receive("ready")
        self._phase = "ready"
        return WorkerConnection(self.expected.scope, self._reader, self._writer)

    def close(self) -> None:
        if self._phase == "transferred":
            return
        self._phase = "closed"
        try:
            self._writer.close()
        finally:
            self._reader.close()


def start_workers(
    workers: Mapping[int, WorkerStartupConnection], *, stop: Callable[[], None]
) -> WorkerGroup:
    """Check every CPU preflight, release every start, then await all readiness.

    ``stop`` terminates only the node-daemon-owned processes. A partial startup failure
    closes every stream before that callback. No TrainingClient may be published
    until this barrier returns; there is no training state to resume on failure.
    """
    # A previous successful call transferred ownership to its live group. A
    # repeated API call must not stop that group's already-owned processes.
    for worker in workers.values():
        worker._require("new")
    with ExitStack() as cleanup:
        cleanup.callback(stop)
        for worker in workers.values():
            cleanup.callback(worker.close)
        if not workers:
            raise ValueError("startup requires an assigned worker set")
        ordered = dict(sorted(workers.items()))
        first = next(iter(ordered.values())).expected
        if ordered.keys() != set(first.logical_by_worker):
            raise ValueError("startup participants differ from the assigned worker set")
        for rank, worker in ordered.items():
            item = worker.expected
            if (
                rank != item.scope.rank
                or item.scope.run_id != first.scope.run_id
                or item.scope.server_incarnation != first.scope.server_incarnation
                or item.logical_by_worker != first.logical_by_worker
                or item.bundle != first.bundle
            ):
                raise ValueError(
                    "startup workers disagree on run, bundle or assignment"
                )
        for worker in ordered.values():
            worker.prepared()
        for worker in ordered.values():
            worker.release()
        group = WorkerGroup(
            {rank: worker.ready() for rank, worker in ordered.items()}, stop=stop
        )
        for worker in ordered.values():
            worker._phase = "transferred"
        cleanup.pop_all()
        return group


def await_start(expected: WorkerStartup, reader: BinaryIO, writer: BinaryIO) -> None:
    """Worker preflight boundary; call before importing/opening GPU execution."""
    write_message(writer, WorkerBoot(expected, "prepared"))
    if read_message(reader) != WorkerBoot(expected, "start"):
        raise ProtocolError("worker did not receive its exact admitted startup release")


def report_ready(expected: WorkerStartup, writer: BinaryIO) -> None:
    """Publish only after the admitted Executor and initialization both succeed."""
    write_message(writer, WorkerBoot(expected, "ready"))
