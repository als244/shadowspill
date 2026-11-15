"""Disposable bounded index of durable run descriptions, scoped on every query."""

import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory

from shadowspill_client.protocol.codec import decode, encode
from shadowspill_client.protocol.runs import RunInfo

from .requests import RunIdentity

REGISTRATION_BYTES = 8192
# Preserve the registration allowance and leave room for a 4 KiB Plan reference,
# a 128-character history ID and the complete ServiceError, whose 1,024 Unicode
# characters can require 12 JSON bytes each. Caller-owned pages remain bounded.
RUN_BYTES = 32 << 10


def registration_bytes(info: RunInfo) -> bytes:
    return encode(info, max_bytes=REGISTRATION_BYTES)


def run_bytes(info: RunInfo) -> bytes:
    return encode(info, max_bytes=RUN_BYTES)


class RunIndex:
    def __init__(self, cache_root: Path) -> None:
        self._directory = TemporaryDirectory(prefix="run-index-", dir=cache_root)
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
            self._db.execute("""CREATE TABLE runs (
                owner TEXT NOT NULL, session TEXT NOT NULL, run TEXT NOT NULL,
                position INTEGER UNIQUE NOT NULL, incarnation TEXT NOT NULL,
                state TEXT NOT NULL, info BLOB NOT NULL,
                PRIMARY KEY (owner, session, run))""")
        except BaseException:
            if hasattr(self, "_db"):
                self._db.close()
            self._directory.cleanup()
            raise

    @staticmethod
    def _decode(raw: bytes) -> RunInfo:
        value = decode(raw, max_bytes=RUN_BYTES)
        if not isinstance(value, RunInfo):
            raise ValueError("invalid derived run description")
        return value

    def get(self, identity: RunIdentity) -> RunInfo | None:
        row = self._db.execute(
            "SELECT info FROM runs WHERE owner=? AND session=? AND run=?",
            (identity.owner_id, identity.session_id, identity.run_id),
        ).fetchone()
        if row is None:
            return None
        info = self._decode(row[0])
        if (info.owner_id, info.session_id, info.run_id) != (
            identity.owner_id,
            identity.session_id,
            identity.run_id,
        ):
            raise ValueError("derived run description differs from its scope")
        return info

    def insert(self, position: int, info: RunInfo) -> None:
        self._db.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                info.owner_id,
                info.session_id,
                info.run_id,
                position,
                info.server_incarnation,
                info.state,
                run_bytes(info),
            ),
        )

    def update(self, info: RunInfo) -> None:
        self._db.execute(
            "UPDATE runs SET state=?, info=? WHERE owner=? AND session=? AND run=?",
            (info.state, run_bytes(info), info.owner_id, info.session_id, info.run_id),
        )

    def page(
        self, owner: str, session: str, after: str, limit: int
    ) -> tuple[RunInfo, ...]:
        return tuple(
            self._decode(row[0])
            for row in self._db.execute(
                "SELECT info FROM runs WHERE owner=? AND session=? AND run>? "
                "ORDER BY run LIMIT ?",
                (owner, session, after, limit),
            )
        )

    def interrupted(
        self, incarnation: str, after: int, limit: int
    ) -> tuple[tuple[int, RunInfo], ...]:
        return tuple(
            (int(row[0]), self._decode(row[1]))
            for row in self._db.execute(
                "SELECT position, info FROM runs WHERE incarnation<>? AND position>? "
                "AND state IN ('preparing', 'ready') ORDER BY position LIMIT ?",
                (incarnation, after, limit),
            )
        )

    def close(self) -> None:
        self._db.close()
        self._directory.cleanup()
