"""Stage all participants, then release all before waiting on collective work."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from threading import Lock, RLock
from types import MappingProxyType
from typing import Self

from shadowspill.training.session.dispatch import ExecutionNotStarted
from shadowspill.training.session.prepared import PreparedCall
from shadowspill.training.session.state import Transition

from .bindings import worker_call
from .connection import WorkerConnection
from .protocol import WorkerCall, WorkerReply


class WorkerGroup:
    """One exclusive set of live worker connections owned by allocation agents.

    Agents finish bounded file delivery before these commands and supply ``stop``
    to terminate all owned children on close/uncertain failure. This controller
    does no packing, GPU work, training state transition or automatic retry.
    """

    def __init__(
        self, connections: Mapping[int, WorkerConnection], *, stop: Callable[[], None]
    ) -> None:
        if not connections or any(
            rank != item.scope.rank for rank, item in connections.items()
        ):
            raise ValueError("worker connections must match their logical ranks")
        if (
            len(
                {
                    (item.scope.run_id, item.scope.server_incarnation)
                    for item in connections.values()
                }
            )
            != 1
        ):
            raise ValueError("worker connections must belong to one live run")
        self.connections = MappingProxyType(dict(sorted(connections.items())))
        self._stop = stop
        self._busy = Lock()
        self._close_lock = RLock()
        self.valid = True
        self.closed = False

    def execute(
        self, call: PreparedCall, transition: Transition
    ) -> Mapping[int, WorkerReply]:
        """Adapt an accepted prepared call after agents finish file delivery."""
        return self.run(self.commands(call, transition))

    def commands(
        self, call: PreparedCall, transition: Transition
    ) -> Mapping[int, WorkerCall]:
        """Resolve the next invocation before agents reserve/deliver its files."""
        self._available()
        try:
            commands = {
                rank: worker_call(
                    call, transition, connection.scope, connection.next_invocation
                )
                for rank, connection in self.connections.items()
            }
            if call.worker_plans(transition).keys() != self.connections.keys():
                raise ValueError("prepared call requires a different worker set")
        except (TypeError, ValueError, RuntimeError) as error:
            raise ExecutionNotStarted(str(error)) from error
        return MappingProxyType(commands)

    def run(self, commands: Mapping[int, WorkerCall]) -> Mapping[int, WorkerReply]:
        """Return all confirmed effects, including host-publication failures.

        The dispatcher records completion before its publisher handles a failed
        output reply. Unknown execution/lost replies still stop the whole group.
        """
        with self.staged(commands) as invocation:
            invocation.execute()
            for rank in self.connections:
                invocation.finish(rank)
            return invocation.results

    @contextmanager
    def staged(self, commands: Mapping[int, WorkerCall]) -> Iterator[WorkerInvocation]:
        """Pause after every ready reply so the owner can admit complete outputs.

        Resource admission failure here aborts all ready workers and confirms
        unstarted execution. Once released, any unconsumed or unknown completion
        stops the group. Ordinary run() uses this same boundary without a pause.
        """
        if not self._busy.acquire(blocking=False):
            raise RuntimeError("another worker-group invocation is in flight")
        invocation = None
        try:
            self._available()
            # All local metadata is checked before any worker receives a command.
            if commands.keys() != self.connections.keys():
                raise ExecutionNotStarted(
                    "command participants differ from the worker group"
                )
            try:
                for rank, connection in self.connections.items():
                    connection.check(commands[rank])
            except (TypeError, ValueError) as error:
                raise ExecutionNotStarted(str(error)) from error
            try:
                for rank, connection in self.connections.items():
                    connection.stage(commands[rank])
                staged = {
                    rank: connection.staged()
                    for rank, connection in self.connections.items()
                }
                failed = [
                    (rank, reply)
                    for rank, reply in staged.items()
                    if reply.status == "failed"
                ]
                if failed:
                    ready = [
                        self.connections[rank]
                        for rank, reply in staged.items()
                        if reply.status == "ready"
                    ]
                    for connection in ready:
                        connection.release("abort")
                    for connection in ready:
                        connection.finished()
                    rank, reply = failed[0]
                    raise ExecutionNotStarted(
                        f"Worker {rank} rejected input staging: {reply.message}"
                    )
                invocation = WorkerInvocation(self.connections, staged)
                try:
                    yield invocation
                except Exception as error:
                    if not invocation.released:
                        invocation._abort()
                        raise ExecutionNotStarted(str(error)) from error
                    if isinstance(error, ExecutionNotStarted):
                        raise RuntimeError(
                            "released workers cannot claim unstarted execution"
                        ) from error
                    raise
                if not invocation.released:
                    invocation._abort()
                    raise ExecutionNotStarted("staged invocation was not released")
                if not invocation.complete:
                    raise RuntimeError("released workers have unconsumed completions")
            except ExecutionNotStarted:
                raise
            except BaseException:
                self.close()
                raise
        finally:
            if invocation is not None:
                invocation._active = False
            self._busy.release()

    def _available(self) -> None:
        if not self.valid or not all(item.valid for item in self.connections.values()):
            self.close()
            raise RuntimeError("worker group is invalid; never replay execution")

    def close(self) -> None:
        with self._close_lock:
            if not self.closed:
                self.valid = False
                with ExitStack() as cleanup:
                    cleanup.callback(self._stop)
                    for connection in self.connections.values():
                        cleanup.callback(connection.close)
                self.closed = True

    def __enter__(self) -> Self:
        if not self.valid:
            raise RuntimeError("worker group is closed")
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class WorkerInvocation:
    """One staged group, borrowed only inside WorkerGroup.staged().

    Release is broadcast before consuming any participant. The caller may receive
    an admitted binary stream from a connection before finish(rank); its matching
    ordinary reply still establishes completion. No command is ever resent.
    """

    def __init__(
        self,
        connections: Mapping[int, WorkerConnection],
        ready: Mapping[int, WorkerReply],
    ) -> None:
        self.connections = connections
        self.ready = MappingProxyType(dict(ready))
        self.released = False
        self._active = True
        self._completed: dict[int, WorkerReply] = {}

    @property
    def complete(self) -> bool:
        return self._completed.keys() == self.connections.keys()

    @property
    def results(self) -> Mapping[int, WorkerReply]:
        if not self.complete:
            raise RuntimeError("worker invocation has incomplete results")
        return MappingProxyType(dict(self._completed))

    def execute(self) -> None:
        if not self._active or self.released:
            raise RuntimeError("worker invocation is retired or already released")
        # Set before the first send: a partial release is never an unstarted abort.
        self.released = True
        for connection in self.connections.values():
            connection.release("execute")

    def finish(self, rank: int) -> WorkerReply:
        if not self._active or not self.released or rank in self._completed:
            raise RuntimeError(
                "worker completion is retired, unstarted or already read"
            )
        reply = self.connections[rank].finished()
        if reply.effect != "completed":
            raise RuntimeError(f"Worker {rank} failed after release: {reply.message}")
        self._completed[rank] = reply
        return reply

    def _abort(self) -> None:
        if not self._active or self.released:
            raise RuntimeError("cannot abort a retired or released invocation")
        for connection in self.connections.values():
            connection.release("abort")
        for connection in self.connections.values():
            connection.finished()
