"""Coordinate durable writer fencing with old upload rejection and sequence holes."""

from dataclasses import dataclass
from typing import Protocol

from shadowspill_client.protocol.requests import WriterLease

from .acceptance import AcceptanceOrder
from .inputs import InputRecords, PreparedInput, RequestInputs


class WriterRecords(InputRecords, Protocol):
    @property
    def writer(self) -> WriterLease | None: ...

    def claim_writer(self, writer_id: str, server_incarnation: str) -> WriterLease: ...


@dataclass(frozen=True)
class Attachment:
    lease: WriterLease
    ready: bool


class WriterOwnership[Data: PreparedInput]:
    """A writer transfer never closes accepted work or frees active input owners.

    New metadata stays gated while old rejected uploads wait for preceding
    execution to settle. This prevents a new writer's explicit corrected calls
    from being included in the old writer's dependency-failure suffix. The
    service checks status after completion or on an idempotent attach retry.
    This transfers authority only; it does not initialize/restore lost workers.
    """

    def __init__(
        self,
        records: WriterRecords,
        order: AcceptanceOrder,
        inputs: RequestInputs[Data],
    ) -> None:
        self._records, self._order, self._inputs = records, order, inputs
        self._pending: WriterLease | None = None

    def attach(self, writer_id: str, server_incarnation: str) -> Attachment:
        with self._order.synchronized():
            try:
                prior = self._records.writer
                lease = self._records.claim_writer(writer_id, server_incarnation)
            except (PermissionError, ValueError):
                # Expected retired identities cannot take ownership back. A
                # broken store instead fails this query and stops live work.
                try:
                    _ = self._records.writer
                except BaseException:
                    self._order.unavailable()
                    raise
                raise
            except BaseException:
                self._order.unavailable()
                raise
            if lease != prior:
                self._inputs.pause_registration(True)
                self._pending = lease
                try:
                    self._order.retire_before(lease.next_sequence)
                    self._inputs.staging.retire_before(lease.next_sequence)
                    self._reject_uploads(lease)
                except BaseException:
                    self._order.unavailable()
                    raise
            return self.status(writer_id)

    def status(self, writer_id: str) -> Attachment:
        with self._order.synchronized():
            try:
                lease = self._records.writer
                if lease is None or lease.writer_id != writer_id:
                    raise PermissionError("writer attachment was fenced")
                if self._pending is None:
                    return Attachment(lease, True)
                self._order.advance()
                self._inputs.reap()
                after = 0
                while page := self._records.pending(after=after, limit=128):
                    if any(status.state == "uploading" for status in page):
                        return Attachment(lease, False)
                    after = page[-1].identity.sequence
                self._pending = None
                self._inputs.pause_registration(False)
                return Attachment(lease, True)
            except PermissionError:
                raise
            except BaseException:
                self._order.unavailable()
                raise

    def _reject_uploads(self, lease: WriterLease) -> None:
        after = 0
        while page := self._records.pending(after=after, limit=128):
            for status in page:
                if status.state == "uploading" and not lease.permits(status.identity):
                    self._inputs.reject(
                        status.identity.request_id,
                        code="writer_fenced",
                        message="The upload belongs to a replaced writer.",
                    )
                after = status.identity.sequence
