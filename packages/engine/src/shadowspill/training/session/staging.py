"""Reserve complete input pipelines in submission order, before upload/copying."""

from __future__ import annotations

from threading import Condition
from time import monotonic

from shadowspill.training.inputs.staging import StagingReservation
from shadowspill_client import Execution
from shadowspill_client._records import integer, number
from shadowspill_client.buffers import BufferBudget


class StagingTicket:
    """One ownership transfer from queue to upload/validation/execution producer.

    A timeout does not cancel the ticket. After acquisition, its owner must close
    all inner payloads before closing the ticket. Fencing may mark a request
    failed, but must not release a producer's storage while it is still copying.
    """

    def __init__(self, queue: OrderedStaging, sequence: int, size: int) -> None:
        self.sequence, self.size = sequence, size
        self._queue = queue
        self._reservation: StagingReservation | None = None
        self._claimed = self._closed = False

    def acquire(self, *, timeout: float | None = None) -> BufferBudget:
        if timeout is not None:
            number(timeout, "staging wait timeout", minimum=0)
        deadline = None if timeout is None else monotonic() + timeout
        with self._queue._condition:
            while True:
                if self._closed or self._queue._closed:
                    raise RuntimeError("input staging ticket is closed")
                if self._claimed:
                    raise RuntimeError("input staging already has an owner")
                if self._reservation is not None:
                    self._claimed = True
                    return self._reservation.budget
                remaining = None if deadline is None else deadline - monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("waiting for ordered input staging")
                self._queue._condition.wait(remaining)

    def close(self) -> None:
        with self._queue._condition:
            if self._closed:
                return
            if self._reservation is not None:
                self._reservation.close()  # Refuses release while inner owners live.
            self._closed = True
            self._queue._finish(self.sequence)

    def cancel_wait(self) -> None:
        """Wake a rejected waiting producer; never release an acquired arena."""
        with self._queue._condition:
            if not self._claimed:
                self.close()

    def __enter__(self) -> BufferBudget:
        return self.acquire()

    def __exit__(self, *exc: object) -> None:
        self.close()


class OrderedStaging:
    """Bounded metadata window plus the existing admitted input BufferBudget.

    Later uploads cannot consume space before earlier pipeline reservations are
    granted. Once granted, independent producers may finish in any order. Full
    windows return backpressure without occupying a waiting HTTP handler. The
    request coordinator supplies every sequence, including a one-byte slot for
    tensor-free commands, and closes rejected/expired unclaimed tickets.
    """

    def __init__(
        self,
        execution: Execution,
        *,
        first_sequence: int = 1,
        budget: BufferBudget | None = None,
    ) -> None:
        self._first = self._next = integer(first_sequence, "first sequence", minimum=1)
        self._maximum = execution.max_pending_requests
        if budget is not None and (
            budget.max_bytes != execution.max_staging_bytes
            or budget.max_requests != self._maximum
            or budget.used != (0, 0)
        ):
            raise ValueError(
                "input staging requires an unused budget matching Execution"
            )
        # An allocation service supplies its exclusive node-backed arena here.
        # Standalone owners use the same queue with a private buffer budget.
        self._budget = budget or BufferBudget(
            execution.max_staging_bytes, self._maximum
        )
        self._condition = Condition()
        self._entries: dict[int, StagingTicket] = {}
        self._finished: set[int] = set()
        self._closed = False

    @property
    def used(self) -> tuple[int, int]:
        return self._budget.used

    def register(self, sequence: int, staging_bytes: int) -> StagingTicket:
        """Register an admitted pipeline bound; allocate no input tensor bytes.

        The size comes from the resolved input-staging description, not an
        untrusted upload's byte count. Each granted arena enforces that bound.
        """
        with self._condition:
            self.preflight(sequence, staging_bytes)
            ticket = StagingTicket(self, sequence, staging_bytes)
            self._entries[sequence] = ticket
            self._grant()
            return ticket

    def preflight(self, sequence: int, staging_bytes: int) -> None:
        """Check metadata capacity before the coordinator durably registers an ID.

        The coordinator serializes registrations. Concurrent owner cleanup can
        only enlarge this window; it cannot consume the missing sequence.
        """
        integer(sequence, "submission sequence", minimum=1)
        integer(staging_bytes, "complete pipeline bytes", minimum=1)
        if staging_bytes > self._budget.max_bytes:
            raise ValueError("complete input pipeline exceeds the staging budget")
        with self._condition:
            if self._closed:
                raise RuntimeError("input staging is closed")
            if (
                sequence < self._first
                or sequence in self._entries
                or sequence in self._finished
            ):
                raise ValueError("submission sequence already owns or consumed a slot")
            if sequence >= self._first + self._maximum:
                raise TimeoutError("ordered submission window is full")

    def _grant(self) -> None:
        # The condition is held. Each reservation is nonblocking; a later input
        # can never consume the space needed to continue an earlier pipeline.
        while not self._closed:
            if self._next in self._finished:
                self._next += 1
                continue
            ticket = self._entries.get(self._next)
            if ticket is None:
                break
            try:
                ticket._reservation = StagingReservation(
                    self._budget, ticket.size, timeout=0
                )
            except TimeoutError:
                break
            self._next += 1
        self._condition.notify_all()

    def retire_before(self, sequence: int) -> None:
        """After a durable writer fence, skip unsent holes, retaining live tickets."""
        integer(sequence, "writer's next sequence", minimum=1)
        with self._condition:
            if sequence > self._first + self._maximum:
                raise ValueError("writer retirement exceeds the bounded input window")
            self._finished.update(
                index
                for index in range(self._first, sequence)
                if index not in self._entries
            )
            while self._first in self._finished:
                self._finished.remove(self._first)
                self._first += 1
            self._next = max(self._next, self._first)
            self._grant()

    def _finish(self, sequence: int) -> None:
        self._entries.pop(sequence)
        self._finished.add(sequence)
        while self._first in self._finished:
            self._finished.remove(self._first)
            self._first += 1
        self._next = max(self._next, self._first)
        self._grant()

    def close(self) -> None:
        """Wake waiters; acquired producers retain storage until their own close."""
        with self._condition:
            self._closed = True
            self._budget.close()
            for ticket in tuple(self._entries.values()):
                if not ticket._claimed:
                    ticket.close()
            self._condition.notify_all()
