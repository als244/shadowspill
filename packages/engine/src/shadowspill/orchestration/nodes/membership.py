"""Agent-side lease and command delivery; training semantics stay in handlers."""

from __future__ import annotations

import http.client
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from threading import Event, Lock, Thread
from time import monotonic

from shadowspill_client._records import Value

from .protocol import (
    Acknowledged,
    Command,
    Heartbeat,
    Join,
    Joined,
    Leave,
    Poll,
    Receipt,
)
from .registry import NodeUnavailable
from .transport import NodeTransport


class Membership:
    """Maintain liveness independently of long commands, logs and bulk transfers.

    A failed/expired membership never rejoins itself and never repeats an unknown
    command. The local resource owner stops all children before acknowledging
    Leave. An external supervisor must cover process death as well as this normal
    shutdown path.
    """

    def __init__(self, transport: NodeTransport, *, max_commands: int = 32) -> None:
        if type(max_commands) is not int or max_commands < 1:
            raise ValueError("command receipt capacity must be positive")
        self.transport, self.max_commands = transport, max_commands
        self.stop = Event()
        self._lock = Lock()
        self._deadline = 0.0
        self._failure: BaseException | None = None
        self.joined: Joined | None = None

    def _active(self) -> None:
        with self._lock:
            if self._failure is not None:
                raise NodeUnavailable("node membership failed") from self._failure
            if self.stop.is_set():
                raise NodeUnavailable("node membership is stopping")
            if monotonic() >= self._deadline:
                raise NodeUnavailable("node lease expired")

    def _heartbeats(self, assignments: Callable[[], Sequence[str]]) -> None:
        assert self.joined is not None
        joined, sequence = self.joined, 0
        try:
            while not self.stop.wait(joined.heartbeat_seconds):
                self._active()
                sent = monotonic()
                sequence += 1
                try:
                    response = self.transport.exchange(
                        "heartbeat", Heartbeat(joined.scope, sequence, assignments())
                    )
                except (OSError, http.client.HTTPException):
                    self._active()
                    continue
                if response != Acknowledged(joined.scope, sequence):
                    raise NodeUnavailable(
                        "heartbeat acknowledgement has the wrong scope"
                    )
                with self._lock:
                    # Bound authority by request-send time, not response-arrival
                    # time. An acknowledgement cannot revive an expired lease.
                    if monotonic() >= self._deadline:
                        raise NodeUnavailable("lease expired during heartbeat exchange")
                    self._deadline = sent + joined.lease_seconds
        except BaseException as error:
            with self._lock:
                self._failure = error
            self.stop.set()

    def run(
        self,
        request: Join,
        *,
        handle: Callable[[Command], Mapping[str, Value]],
        cleanup: Callable[[], None],
        assignments: Callable[[], Sequence[str]],
        joined: Callable[[Joined], None],
    ) -> None:
        sent = monotonic()
        # Joining the same incarnation is idempotent. A lost response may be
        # retried, but neither a retry nor a delayed reply extends its lease.
        for attempt in range(3):
            try:
                response = self.transport.exchange("join", request)
                break
            except (OSError, http.client.HTTPException):
                if attempt == 2 or self.stop.wait(0.1):
                    raise
        if not isinstance(response, Joined):
            raise NodeUnavailable("service omitted node membership")
        self.joined = response
        self._deadline = sent + response.lease_seconds
        heartbeats: Thread | None = None
        receipts: Thread | None = None
        watchdog: Thread | None = None
        cleanup_lock = Lock()
        executor = ThreadPoolExecutor(
            max_workers=self.max_commands, thread_name_prefix="node-command"
        )
        pending: dict[int, tuple[Command, Future[Receipt]]] = {}
        delivered: set[int] = set()
        state_lock, ready = Lock(), Event()
        acknowledged = retired = 0

        def stop_owned() -> None:
            with cleanup_lock:
                cleanup()

        def watch_lease() -> None:
            try:
                while not self.stop.wait(0.05):
                    self._active()
            except BaseException as error:
                with self._lock:
                    self._failure = error
                self.stop.set()
            finally:
                try:
                    stop_owned()
                except BaseException as error:
                    with self._lock:
                        self._failure = error
                    self.stop.set()

        def execute(command: Command) -> Receipt:
            self._active()
            try:
                return Receipt(command.scope, command.sequence, handle(command))
            except Exception as error:
                return Receipt(
                    command.scope,
                    command.sequence,
                    error=(str(error).strip() or type(error).__name__)[:1024],
                )

        def accept_retirement(value: int) -> None:
            nonlocal retired
            if not retired <= value <= acknowledged:
                raise NodeUnavailable(
                    "service returned an invalid retirement watermark"
                )
            retired = value
            for sequence in tuple(pending):
                if sequence <= retired:
                    if not pending[sequence][1].done():
                        raise NodeUnavailable("service retired an incomplete command")
                    del pending[sequence]
                    delivered.discard(sequence)

        def publish() -> None:
            try:
                while not self.stop.is_set():
                    ready.wait(0.1)
                    ready.clear()
                    with state_lock:
                        results = tuple(
                            (command, result)
                            for command, result in pending.values()
                            if result.done() and command.sequence not in delivered
                        )
                    for command, result in results:
                        self._active()
                        try:
                            receipt = self.transport.exchange(
                                "receipt", result.result()
                            )
                        except (OSError, http.client.HTTPException):
                            ready.set()
                            self.stop.wait(0.1)
                            continue
                        if (
                            not isinstance(receipt, Acknowledged)
                            or receipt.scope != response.scope
                        ):
                            raise NodeUnavailable(
                                "receipt acknowledgement has the wrong scope"
                            )
                        with state_lock:
                            if command.sequence > retired:
                                delivered.add(command.sequence)
                            # Another transport lane can advance retirement in
                            # the meantime. An earlier acknowledgement is safe.
                            accept_retirement(max(retired, receipt.sequence))
            except BaseException as error:
                with self._lock:
                    self._failure = error
                self.stop.set()

        try:
            joined(response)
            if response.poll_seconds + 1 >= self.transport.timeout:
                raise ValueError(
                    "node HTTP timeout must exceed negotiated poll duration"
                )
            self._active()
            watchdog = Thread(target=watch_lease, name="node-lease-watchdog")
            watchdog.start()
            heartbeats = Thread(
                target=self._heartbeats, args=(assignments,), name="node-heartbeat"
            )
            heartbeats.start()
            receipts = Thread(target=publish, name="node-receipts")
            receipts.start()
            while not self.stop.is_set():
                self._active()
                try:
                    with state_lock:
                        poll = Poll(response.scope, acknowledged, retired)
                    next_command = self.transport.exchange("poll", poll)
                except (OSError, http.client.HTTPException):
                    self._active()
                    self.stop.wait(0.1)
                    continue
                if self.stop.is_set():
                    break
                if isinstance(next_command, Acknowledged):
                    if next_command.scope != response.scope:
                        raise NodeUnavailable("poll has the wrong membership scope")
                    with state_lock:
                        accept_retirement(max(retired, next_command.sequence))
                    continue
                if (
                    not isinstance(next_command, Command)
                    or next_command.scope != response.scope
                ):
                    raise NodeUnavailable("command has the wrong membership scope")
                with state_lock:
                    if next_command.sequence <= acknowledged:
                        previous = pending.get(next_command.sequence)
                        if previous is None or previous[0] != next_command:
                            raise NodeUnavailable(
                                "command changed or was already retired"
                            )
                        continue
                    if next_command.sequence != acknowledged + 1 or len(
                        pending
                    ) >= self.max_commands + (next_command.operation == "stop"):
                        raise NodeUnavailable(
                            "command order/capacity exceeds membership contract"
                        )
                    # Stop has an independent lane, so blocked phase/worker
                    # commands cannot starve cancellation.
                    if next_command.operation == "stop":
                        completed: Future[Receipt] = Future()
                        completed.set_result(execute(next_command))
                    else:
                        completed = executor.submit(execute, next_command)
                    pending[next_command.sequence] = next_command, completed
                    acknowledged = next_command.sequence
                    completed.add_done_callback(lambda _: ready.set())
            if self._failure is not None:
                raise NodeUnavailable("node membership failed") from self._failure
        finally:
            self.stop.set()
            try:
                stop_owned()
            finally:
                executor.shutdown(wait=True, cancel_futures=True)
                if heartbeats is not None:
                    heartbeats.join()
                if receipts is not None:
                    receipts.join()
                if watchdog is not None:
                    watchdog.join()
            # A failed cleanup above deliberately withholds this acknowledgement.
            # It never authorizes the service to reuse an uncertain GPU claim.
            with suppress(OSError, http.client.HTTPException, NodeUnavailable):
                self.transport.exchange("leave", Leave(response.scope))
