"""Small controller startup log, owned by the existing creation host envelope."""

import hashlib
import io
from threading import RLock

from shadowspill.diagnostics.progress import Progress
from shadowspill_client._records import integer
from shadowspill_client.protocol.progress import LOG_CHUNK_BYTES
from shadowspill_client.protocol.requests import ArtifactRef

from .artifacts import ArtifactStore
from .creation_logs import publish_metadata
from .records.requests import RunIdentity


class StartupLog:
    def __init__(self) -> None:
        self.limit_bytes = 64 << 10
        self._data = bytearray()
        self._lock = RLock()
        self._complete = False
        self.reporter = Progress(self.append)

    def append(self, message: str) -> None:
        data = (message + "\n").encode()
        with self._lock:
            if not self._complete:
                self._data.extend(data[: self.limit_bytes - len(self._data)])

    def finish(self) -> None:
        with self._lock:
            self._complete = True

    def read(self, name: str, offset: int) -> tuple[bytes, bool, bool]:
        integer(offset, "log offset")
        with self._lock:
            data = self._data if name == "stdout" else b""
            if name not in ("stdout", "stderr") or offset > len(data):
                raise ValueError("unknown log stream or offset exceeds available bytes")
            chunk = bytes(data[offset : offset + LOG_CHUNK_BYTES])
            return (
                chunk,
                self._complete and offset + len(chunk) == len(data),
                len(data) == self.limit_bytes,
            )

    def publish(self, store: ArtifactStore, identity: RunIdentity) -> None:
        with self._lock:
            if not self._complete:
                raise RuntimeError("startup log producer is still active")
            data = bytes(self._data)
        digest = hashlib.sha256(data).hexdigest()
        reference = ArtifactRef(
            "startup-log." + digest, digest, len(data), "text/plain"
        )
        store.put(reference, lambda _: io.BytesIO(data))
        publish_metadata(
            store,
            identity,
            "phase:startup",
            {
                "phase": "startup",
                "limit_bytes": self.limit_bytes,
                "streams": {"stdout": reference.to_dict()},
            },
        )
