"""Bounded local query cache rebuilt from the owner's artifact journal."""

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from shadowspill_client._records import Record, text
from shadowspill_client.protocol.codec import RecordCodec
from shadowspill_client.protocol.requests import ArtifactRef, identifier

REFERENCE_BYTES = 4096
_CODEC = RecordCodec({"artifact.reference": ArtifactRef})


def alias_scope(session_id: str, run_id: str, name: str) -> tuple[str, str, str]:
    identifier(session_id, "artifact session ID")
    identifier(run_id, "artifact run ID")
    if len(text(name, "artifact name")) > 256:
        raise ValueError("artifact name exceeds its bound")
    return session_id, run_id, name


@dataclass(frozen=True)
class ArtifactAlias(Record):
    """A run-local friendly name, updated in its generation's durable commit."""

    session_id: str
    run_id: str
    name: str
    artifact_id: str

    def __post_init__(self) -> None:
        alias_scope(self.session_id, self.run_id, self.name)
        identifier(self.artifact_id, "named artifact ID")


def storage_key(artifact_id: str) -> str:
    return hashlib.sha256(artifact_id.encode()).hexdigest()


def reference_bytes(reference: ArtifactRef) -> bytes:
    if len(reference.media_type.encode()) > 256:
        raise ValueError("artifact media type exceeds its bound")
    return _CODEC.encode(reference, max_bytes=REFERENCE_BYTES)


class ArtifactIndex:
    def __init__(self, cache_root: Path) -> None:
        self._directory = TemporaryDirectory(prefix="artifact-index-", dir=cache_root)
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
            self._db.execute(
                "CREATE TABLE artifacts (id TEXT PRIMARY KEY, "
                "storage_key TEXT UNIQUE NOT NULL, reference BLOB NOT NULL)"
            )
            self._db.execute(
                "CREATE TABLE aliases (session_id TEXT, run_id TEXT, name TEXT, "
                "artifact_id TEXT NOT NULL, PRIMARY KEY (session_id, run_id, name))"
            )
        except BaseException:
            if hasattr(self, "_db"):
                self._db.close()
            self._directory.cleanup()
            raise

    @staticmethod
    def _decode(raw: bytes) -> ArtifactRef:
        result = _CODEC.decode(raw, max_bytes=REFERENCE_BYTES)
        if not isinstance(result, ArtifactRef):
            raise ValueError("invalid derived artifact index")
        return result

    def get(self, artifact_id: str) -> ArtifactRef | None:
        row = self._db.execute(
            "SELECT reference FROM artifacts WHERE id=?", (artifact_id,)
        ).fetchone()
        return None if row is None else self._decode(row[0])

    def contains(self, key: str) -> bool:
        return (
            self._db.execute(
                "SELECT 1 FROM artifacts WHERE storage_key=?", (key,)
            ).fetchone()
            is not None
        )

    def insert(self, reference: ArtifactRef) -> None:
        self._db.execute(
            "INSERT INTO artifacts VALUES (?, ?, ?)",
            (reference.id, storage_key(reference.id), reference_bytes(reference)),
        )

    def page(self, after: str, limit: int) -> tuple[ArtifactRef, ...]:
        return tuple(
            self._decode(row[0])
            for row in self._db.execute(
                "SELECT reference FROM artifacts WHERE id > ? ORDER BY id LIMIT ?",
                (after, limit),
            )
        )

    def set_alias(self, alias: ArtifactAlias) -> None:
        if self.get(alias.artifact_id) is None:
            raise ValueError("artifact alias requires a committed target")
        self._db.execute(
            "INSERT OR REPLACE INTO aliases VALUES (?, ?, ?, ?)",
            (alias.session_id, alias.run_id, alias.name, alias.artifact_id),
        )

    def resolve_alias(
        self, session_id: str, run_id: str, name: str
    ) -> ArtifactRef | None:
        row = self._db.execute(
            "SELECT artifact_id FROM aliases "
            "WHERE session_id=? AND run_id=? AND name=?",
            (session_id, run_id, name),
        ).fetchone()
        return None if row is None else self.get(row[0])

    def close(self) -> None:
        self._db.close()
        self._directory.cleanup()
