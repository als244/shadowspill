"""Disposable local query cache; the journal remains the only authority."""

import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory

from shadowspill_client.protocol.codec import decode, digest, encode
from shadowspill_client.protocol.metrics import MetricRecord
from shadowspill_client.protocol.requests import OperationRequest, WriterLease
from shadowspill_client.protocol.results import ExecutionReceipt, RequestStatus


class RequestIndex:
    def __init__(self, cache_root: Path) -> None:
        self._directory = TemporaryDirectory(prefix="request-index-", dir=cache_root)
        try:
            self._db = sqlite3.connect(
                Path(self._directory.name) / "index.sqlite",
                check_same_thread=False,
                isolation_level=None,
            )
            self._db.execute("PRAGMA journal_mode=OFF")
            self._db.execute("PRAGMA synchronous=OFF")
            self._db.execute("PRAGMA cache_size=-2048")
            self._db.execute("PRAGMA mmap_size=0")
            self._db.execute("""CREATE TABLE requests (
                id TEXT PRIMARY KEY, sequence INTEGER UNIQUE NOT NULL,
                digest TEXT NOT NULL, request BLOB NOT NULL, status BLOB NOT NULL,
                state TEXT NOT NULL, execution BLOB, finished_position INTEGER)""")
            self._db.execute(
                "CREATE INDEX completed_requests ON requests(finished_position)"
            )
            self._db.execute(
                "CREATE TABLE writers (id TEXT PRIMARY KEY, lease BLOB NOT NULL)"
            )
            self._db.execute("""CREATE TABLE caller_metrics (
                position INTEGER PRIMARY KEY, request_id TEXT NOT NULL,
                annotation_id TEXT NOT NULL, record BLOB NOT NULL,
                UNIQUE(request_id, annotation_id))""")
        except BaseException:
            if hasattr(self, "_db"):
                self._db.close()
            self._directory.cleanup()
            raise

    def get(
        self, request_id: str
    ) -> tuple[OperationRequest, RequestStatus, str] | None:
        row = self._db.execute(
            "SELECT request, status, digest FROM requests WHERE id=?", (request_id,)
        ).fetchone()
        if row is None:
            return None
        request, status = decode(row[0]), decode(row[1])
        if not isinstance(request, OperationRequest) or not isinstance(
            status, RequestStatus
        ):
            raise ValueError("invalid derived request index")
        if (
            request.identity != status.identity
            or request.identity.request_id != request_id
            or digest(request) != row[2]
        ):
            raise ValueError("derived request index identity/content differs")
        return request, status, str(row[2])

    def sequence_used(self, sequence: int) -> bool:
        return (
            self._db.execute(
                "SELECT 1 FROM requests WHERE sequence=?", (sequence,)
            ).fetchone()
            is not None
        )

    def execution(self, request_id: str) -> ExecutionReceipt | None:
        row = self._db.execute(
            "SELECT execution FROM requests WHERE id=?", (request_id,)
        ).fetchone()
        if row is None or row[0] is None:
            return None
        receipt = decode(row[0])
        if (
            not isinstance(receipt, ExecutionReceipt)
            or receipt.identity.request_id != request_id
        ):
            raise ValueError("invalid derived execution receipt")
        return receipt

    def executed(self, receipt: ExecutionReceipt) -> None:
        self._db.execute(
            "UPDATE requests SET execution=? WHERE id=?",
            (encode(receipt), receipt.identity.request_id),
        )

    def id_at_sequence(self, sequence: int) -> str | None:
        row = self._db.execute(
            "SELECT id FROM requests WHERE sequence=?", (sequence,)
        ).fetchone()
        return None if row is None else str(row[0])

    def writer(self, writer_id: str) -> WriterLease | None:
        row = self._db.execute(
            "SELECT lease FROM writers WHERE id=?", (writer_id,)
        ).fetchone()
        if row is None:
            return None
        lease = decode(row[0])
        if not isinstance(lease, WriterLease) or lease.writer_id != writer_id:
            raise ValueError("invalid derived writer index")
        return lease

    def add_writer(self, lease: WriterLease) -> None:
        self._db.execute(
            "INSERT INTO writers VALUES (?, ?)", (lease.writer_id, encode(lease))
        )

    def insert(
        self, request: OperationRequest, status: RequestStatus, checksum: str
    ) -> None:
        self._db.execute(
            "INSERT INTO requests VALUES (?, ?, ?, ?, ?, ?, NULL, NULL)",
            (
                request.identity.request_id,
                request.identity.sequence,
                checksum,
                encode(request),
                encode(status),
                status.state,
            ),
        )

    def update(self, status: RequestStatus, *, position: int) -> None:
        self._db.execute(
            "UPDATE requests SET status=?, state=?, finished_position=? WHERE id=?",
            (
                encode(status),
                status.state,
                position if status.state in ("completed", "failed") else None,
                status.identity.request_id,
            ),
        )

    def annotation(self, request_id: str, annotation_id: str) -> MetricRecord | None:
        row = self._db.execute(
            "SELECT record FROM caller_metrics WHERE request_id=? AND annotation_id=?",
            (request_id, annotation_id),
        ).fetchone()
        if row is None:
            return None
        value = decode(row[0])
        if (
            not isinstance(value, MetricRecord)
            or value.source != "caller"
            or value.identity.request_id != request_id
            or value.annotation_id != annotation_id
        ):
            raise ValueError("invalid derived caller metric index")
        return value

    def add_annotation(self, row: MetricRecord) -> None:
        self._db.execute(
            "INSERT INTO caller_metrics VALUES (?, ?, ?, ?)",
            (row.position, row.identity.request_id, row.annotation_id, encode(row)),
        )

    def metric_events(
        self, after: int, through: int, limit: int
    ) -> list[tuple[int, str, str | None]]:
        return [
            (int(row[0]), str(row[1]), row[2])
            for row in self._db.execute(
                """SELECT finished_position AS position, id, NULL FROM requests
                WHERE finished_position > ? AND finished_position <= ?
                UNION ALL SELECT position, request_id, annotation_id FROM caller_metrics
                WHERE position > ? AND position <= ?
                ORDER BY position LIMIT ?""",
                (after, through, after, through, limit),
            )
        ]

    def pending(self, after: int, limit: int) -> list[str]:
        return [
            row[0]
            for row in self._db.execute(
                """SELECT id FROM requests WHERE sequence > ?
            AND state IN ('uploading', 'accepted', 'running')
            ORDER BY sequence LIMIT ?""",
                (after, limit),
            )
        ]

    @property
    def last_sequence(self) -> int:
        row = self._db.execute(
            "SELECT COALESCE(MAX(sequence), 0) FROM requests"
        ).fetchone()
        return int(row[0])

    def close(self) -> None:
        self._db.close()
        self._directory.cleanup()
