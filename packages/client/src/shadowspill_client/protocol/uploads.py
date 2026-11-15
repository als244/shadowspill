"""Portable, bounded artifact-upload progress, separate from training requests."""

import hashlib
import re
from dataclasses import dataclass
from typing import Literal

from .._records import Record, integer, number, record
from .requests import ArtifactRef

MAX_UPLOAD_CHUNK_BYTES = 4 << 20
UPLOAD_RECORD_BYTES = 4096


@dataclass(frozen=True)
class ArtifactUpload(Record):
    reference: ArtifactRef

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "reference", record(ArtifactRef, self.reference, "upload artifact")
        )
        media = self.reference.media_type
        if (
            len(media) > 256
            or not media.isascii()
            or any(ord(char) < 32 or ord(char) == 127 for char in media)
        ):
            raise ValueError("upload media type is not bounded HTTP metadata")


@dataclass(frozen=True)
class UploadStatus(ArtifactUpload):
    offset: int
    prefix_sha256: str
    state: Literal["uploading", "complete"]
    expires_at: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        integer(self.offset, "upload committed prefix bytes")
        if self.offset > self.reference.size_bytes:
            raise ValueError("upload prefix exceeds its artifact")
        if (
            not isinstance(self.prefix_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.prefix_sha256) is None
        ):
            raise ValueError("upload prefix requires a SHA-256 checksum")
        if self.state not in ("uploading", "complete"):
            raise ValueError("unknown artifact upload state")
        if self.offset == 0 and self.prefix_sha256 != hashlib.sha256().hexdigest():
            raise ValueError("empty upload prefix requires the empty checksum")
        if (
            self.offset == self.reference.size_bytes
            and self.prefix_sha256 != self.reference.sha256
        ):
            raise ValueError(
                "complete upload bytes differ from their artifact checksum"
            )
        if self.state == "complete":
            if self.offset != self.reference.size_bytes or self.expires_at is not None:
                raise ValueError(
                    "committed artifact requires all bytes and no upload expiry"
                )
        elif self.expires_at is None or not number(
            self.expires_at, "upload expiry", minimum=0
        ):
            raise ValueError("pending upload requires a positive expiry timestamp")
