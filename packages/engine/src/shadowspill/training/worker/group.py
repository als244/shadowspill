"""Stage all participants, then release all before waiting on collective work."""

from collections.abc import Callable, Mapping
from contextlib import ExitStack
from threading import Lock
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
        self.valid = True

    def execute(
        self, call: PreparedCall, transition: Transition
    ) -> Mapping[int, WorkerReply]:
        """Adapt an accepted prepared call after agents finish file delivery."""
        self._available()
        try:
            commands = {
                rank: worker_call(
                    call, transition, connection.scope, connection.next_invocation
                )
                for rank, connection in self.connections.items()
            }
            if call.candidate.workers.keys() != self.connections.keys():
                raise ValueError("prepared call requires a different worker set")
        except (TypeError, ValueError, RuntimeError) as error:
            raise ExecutionNotStarted(str(error)) from error
        return self.run(commands)

    def run(self, commands: Mapping[int, WorkerCall]) -> Mapping[int, WorkerReply]:
        """Return all confirmed effects, including host-publication failures.

        The dispatcher records completion before its publisher handles a failed
        output reply. Unknown execution/lost replies still stop the whole group.
        """
        if not self._busy.acquire(blocking=False):
            raise RuntimeError("another worker-group invocation is in flight")
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
                for connection in self.connections.values():
                    connection.release("execute")
                completed = {}
                for rank, connection in self.connections.items():
                    reply = connection.finished()
                    if reply.effect != "completed":
                        raise RuntimeError(
                            f"Worker {rank} failed after release: {reply.message}"
                        )
                    completed[rank] = reply
                return MappingProxyType(completed)
            except ExecutionNotStarted:
                raise
            except BaseException:
                self.close()
                raise
        finally:
            self._busy.release()

    def _available(self) -> None:
        if not self.valid or not all(item.valid for item in self.connections.values()):
            self.close()
            raise RuntimeError("worker group is invalid; never replay execution")

    def close(self) -> None:
        if self.valid:
            self.valid = False
            with ExitStack() as cleanup:
                cleanup.callback(self._stop)
                for connection in self.connections.values():
                    cleanup.callback(connection.close)

    def __enter__(self) -> Self:
        if not self.valid:
            raise RuntimeError("worker group is closed")
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
