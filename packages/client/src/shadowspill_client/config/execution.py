"""Bounded staging and immutable execution-lifetime settings."""

from collections.abc import Mapping
from dataclasses import dataclass

from .._records import Record, Value, integer, mapping, number


@dataclass(frozen=True)
class Execution(Record):
    seed: int = 0
    max_staging_bytes: int = 256 << 20
    max_pending_requests: int = 2
    idle_timeout_seconds: float | None = None
    nsys: Mapping[str, Value] | None = None

    def __post_init__(self) -> None:
        integer(self.seed, "seed")
        integer(self.max_staging_bytes, "max_staging_bytes", minimum=1)
        integer(self.max_pending_requests, "max_pending_requests", minimum=1)
        if (
            self.idle_timeout_seconds is not None
            and number(self.idle_timeout_seconds, "idle_timeout_seconds") <= 0
        ):
            raise ValueError("idle_timeout_seconds must be positive or None")
        if self.nsys is not None:
            object.__setattr__(self, "nsys", mapping(self.nsys, "nsys"))
