"""Bounded byte cursors for existing creation logs, independent of text encoding."""

from base64 import b64decode
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from .._records import Record, integer, record
from .requests import identifier

LOG_CHUNK_BYTES = 16 << 10
LOG_STREAMS = tuple(
    f"{phase}.{stream}"
    for phase in ("build", "profile", "communicate", "plan", "converting")
    for stream in ("stdout", "stderr")
)


@dataclass(frozen=True)
class ProgressRequest(Record):
    offsets: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.offsets, Mapping) or self.offsets.keys() - set(
            LOG_STREAMS
        ):
            raise ValueError("unknown creation log stream")
        offsets = {
            name: integer(value, "log offset") for name, value in self.offsets.items()
        }
        object.__setattr__(self, "offsets", MappingProxyType(offsets))


@dataclass(frozen=True)
class LogChunk(Record):
    stream: str
    offset: int
    data: str
    complete: bool
    capped: bool = False

    def __post_init__(self) -> None:
        if self.stream not in LOG_STREAMS:
            raise ValueError("unknown creation log stream")
        integer(self.offset, "log offset")
        if type(self.complete) is not bool or type(self.capped) is not bool:
            raise TypeError("log completion/cap flags must be boolean")
        if not isinstance(self.data, str) or len(self.data) > 4 * (
            (LOG_CHUNK_BYTES + 2) // 3
        ):
            raise ValueError("log chunk exceeds its byte bound")
        if len(self.bytes()) > LOG_CHUNK_BYTES:
            raise ValueError("log chunk exceeds its byte bound")

    def bytes(self) -> bytes:
        return b64decode(self.data, validate=True)


@dataclass(frozen=True)
class ProgressPage(Record):
    session_id: str
    run_id: str
    chunks: Sequence[LogChunk]

    def __post_init__(self) -> None:
        identifier(self.session_id, "session ID")
        identifier(self.run_id, "run ID")
        if not isinstance(self.chunks, (list, tuple)) or len(self.chunks) > len(
            LOG_STREAMS
        ):
            raise ValueError("creation progress has too many streams")
        chunks = tuple(record(LogChunk, chunk, "log chunk") for chunk in self.chunks)
        names = [chunk.stream for chunk in chunks]
        if len(set(names)) != len(names):
            raise ValueError("creation progress repeats a stream")
        object.__setattr__(self, "chunks", chunks)
