"""Owned input upload and preprocessing lifetime around durable request ordering."""

from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Event
from typing import Protocol

from shadowspill_client import Execution
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.payload import BatchSnapshot
from shadowspill_client.protocol.payload_io import BufferReader, read_payload
from shadowspill_client.protocol.requests import ComputeRequest, identifier
from shadowspill_client.protocol.results import RequestStatus

from .acceptance import AcceptanceOrder, AcceptanceRecords
from .staging import OrderedStaging, StagingTicket


class InputRecords(AcceptanceRecords, Protocol):
    def submit(self, request: ComputeRequest) -> RequestStatus: ...


class PreparedInput(Protocol):
    """Preprocessing finished for the whole call; close releases its owners."""

    def close(self) -> None: ...


@dataclass
class _Slot[Data: PreparedInput]:
    request: ComputeRequest
    ticket: StagingTicket
    cancelled: Event = field(default_factory=Event)
    producing: bool = False
    taken: bool = False
    snapshot: BatchSnapshot | None = None
    data: Data | None = None

    def close(self) -> None:
        # Refuse to release an arena while a producer or execution owner uses it.
        if self.producing or self.taken:
            raise RuntimeError("input request still has an active owner")
        if self.data is not None:
            self.data.close()
            self.data = None
        if self.snapshot is not None:
            self.snapshot.close()
            self.snapshot = None
        self.ticket.close()


class _Reader:
    def __init__(self, stream: BufferReader, cancelled: Event) -> None:
        self.stream, self.cancelled = stream, cancelled

    def readinto(self, buffer: memoryview, /) -> int | None:
        if self.cancelled.is_set():
            raise RuntimeError("input upload was rejected or fenced")
        size = self.stream.readinto(buffer)
        if self.cancelled.is_set():
            raise RuntimeError("input upload was rejected or fenced")
        return size


class RequestInputs[Data: PreparedInput]:
    """One bounded input slot until execution or producer cleanup finishes.

    The installed preprocessing callback validates the whole call and returns
    owned prepared inputs. It must perform all fallible content preprocessing
    before returning. HTTP/body deadlines belong to the transport. A cancelled
    producer retains storage until its read/callback returns; no other thread
    closes its buffers while it is using them.
    """

    def __init__(
        self,
        records: InputRecords,
        order: AcceptanceOrder,
        execution: Execution,
        prepare: Callable[[ComputeRequest, BatchSnapshot | None, BufferBudget], Data],
        *,
        first_sequence: int = 1,
    ) -> None:
        self._records, self._order, self._prepare = records, order, prepare
        self.staging = OrderedStaging(execution, first_sequence=first_sequence)
        self._slots: dict[str, _Slot[Data]] = {}

    def register(self, request: ComputeRequest, *, staging_bytes: int) -> RequestStatus:
        """Register bounded metadata, using a trusted prepared pipeline byte bound."""
        with self._order.synchronized():
            if self._get(request.identity.request_id) is not None:
                return self._submit(request)  # Checks digest; never recopy/replay.
            self.reap()
            self.staging.preflight(request.identity.sequence, staging_bytes)
            status = self._submit(request)
            try:
                ticket = self.staging.register(request.identity.sequence, staging_bytes)
            except BaseException:
                self._order.unavailable()
                raise
            self._slots[request.identity.request_id] = _Slot(request, ticket)
            return status

    def receive(
        self,
        request_id: str,
        stream: BufferReader | None = None,
        *,
        timeout: float | None = 0,
    ) -> RequestStatus:
        """Decode/checksum and preprocess before ordered acceptance; never dispatch."""
        with self._order.synchronized():
            status = self._status(request_id)
            slot = self._slots.get(request_id)
            if status.state != "uploading" or slot is None or slot.data is not None:
                return status
            if slot.producing:
                raise RuntimeError("this input upload already has a producer")
            slot.producing = True
        # A wait cannot hold the ordering lock needed by earlier owner cleanup.
        try:
            arena = slot.ticket.acquire(timeout=timeout)
        except TimeoutError:
            with self._order.synchronized():
                slot.producing = False
                self.reap()
            raise
        except BaseException as error:
            status = self._failed(slot, error)
            if not isinstance(error, Exception):
                raise
            return status
        try:
            reference = slot.request.payload
            if reference is not None:
                if stream is None:
                    raise ValueError(
                        "this request requires its complete tensor payload"
                    )
                slot.snapshot = read_payload(
                    _Reader(stream, slot.cancelled), reference, arena
                )
                if len(slot.snapshot.datums) != slot.request.datum_count:
                    raise ValueError(
                        "payload Datum count differs from request metadata"
                    )
            elif stream is not None:
                raise ValueError("tensor-free request cannot carry an input payload")
            slot.data = self._prepare(slot.request, slot.snapshot, arena)
            with self._order.synchronized():
                slot.producing = False
                if not slot.cancelled.is_set():
                    self._order.validated(request_id)
                self.reap()
                return self._status(request_id)
        except BaseException as error:
            status = self._failed(slot, error)
            if not isinstance(error, Exception):
                raise
            return status

    def _failed(self, slot: _Slot[Data], error: BaseException) -> RequestStatus:
        key = slot.request.identity.request_id
        with self._order.synchronized():
            slot.producing = False
            try:
                if self._status(key).state == "uploading":
                    self._order.reject(
                        key,
                        code="input_failed",
                        message=str(error) or type(error).__name__,
                    )
            finally:
                self._close(slot)
                self._slots.pop(key, None)
            self.reap()
            return self._status(key)

    def take(self, request_id: str) -> Data:
        """Transfer prepared input use once, after acceptance; caller must finish."""
        with self._order.synchronized():
            if self._status(request_id).state not in ("accepted", "running"):
                raise RuntimeError("only accepted inputs can reach a worker")
            slot = self._slots[request_id]
            if slot.producing or slot.taken or slot.data is None:
                raise RuntimeError("prepared input ownership is unavailable")
            slot.taken = True
            return slot.data

    def finish(self, request_id: str) -> None:
        """Worker input consumers have finished; release all complete-pipeline bytes."""
        with self._order.synchronized():
            slot = self._slots[request_id]
            if not slot.taken:
                raise RuntimeError("input request has no execution owner")
            slot.taken = False
            self._close(slot)
            del self._slots[request_id]
            self.reap()

    def reject(self, request_id: str, *, code: str, message: str) -> RequestStatus:
        """Reject an incomplete upload, preserving active producer ownership."""
        with self._order.synchronized():
            status = self._status(request_id)
            if status.state == "uploading":
                slot = self._slots.get(request_id)
                if slot is not None:
                    slot.cancelled.set()
                    slot.ticket.cancel_wait()
                status = self._order.reject(request_id, code=code, message=message)
            self.reap()
            return status

    def reap(self) -> None:
        """After ordered outcomes, cancel failed producers and release idle owners."""
        with self._order.synchronized():
            for key, slot in tuple(self._slots.items()):
                if self._status(key).state == "failed" or slot.cancelled.is_set():
                    slot.cancelled.set()
                    slot.ticket.cancel_wait()
                    if not slot.producing and not slot.taken:
                        self._close(slot)
                        del self._slots[key]

    def abort(self) -> None:
        """Fatal shutdown without querying unavailable storage; never user cancellation.

        Idle owners release immediately. Active producers and worker consumers
        retain their arenas until their own completion/cleanup, even on failure.
        """
        with self._order.synchronized():
            self._order.unavailable()
            for key, slot in tuple(self._slots.items()):
                slot.cancelled.set()
                slot.ticket.cancel_wait()
                if not slot.producing and not slot.taken:
                    self._close(slot)
                    del self._slots[key]

    def _status(self, request_id: str) -> RequestStatus:
        row = self._get(request_id)
        if row is None:
            raise KeyError(f"unknown request: {request_id}")
        return row[1]

    def _get(self, request_id: str) -> tuple[ComputeRequest, RequestStatus] | None:
        identifier(request_id, "request ID")
        try:
            return self._records.get(request_id)
        except BaseException:
            self._order.unavailable()
            raise

    def _submit(self, request: ComputeRequest) -> RequestStatus:
        try:
            return self._records.submit(request)
        except (PermissionError, ValueError):
            # Expected fencing/content errors leave a usable store. A failed
            # authoritative commit can raise the same type but poisons queries.
            self._get(request.identity.request_id)
            raise
        except BaseException:
            self._order.unavailable()
            raise

    def _close(self, slot: _Slot[Data]) -> None:
        try:
            slot.close()
        except BaseException:
            self._order.unavailable()
            raise
