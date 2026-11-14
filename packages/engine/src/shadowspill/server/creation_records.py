"""Durable creation metadata in the authenticated owner's artifact namespace."""

import hashlib
import io
from dataclasses import dataclass

from shadowspill_client.protocol.codec import MAX_METADATA_BYTES, decode, encode
from shadowspill_client.protocol.creation import TrainingCreation
from shadowspill_client.protocol.requests import ArtifactRef
from shadowspill_client.protocol.runs import RunInfo
from shadowspill_client.protocol.service import CONTROL_MEDIA_TYPE

from .artifacts import ArtifactStore


def _reference(kind: str, body: bytes) -> ArtifactRef:
    digest = hashlib.sha256(body).hexdigest()
    return ArtifactRef(f"{kind}.{digest}", digest, len(body), CONTROL_MEDIA_TYPE)


@dataclass(frozen=True)
class CreationRecord:
    """Bounded metadata, charged to the caller's existing control/creation budget.

    The complete creation digest is also its deterministic artifact key. The run
    catalog keeps that digest plus the separately downloadable resolved config.
    All choices survive the initial HTTP exchange; a factory never depends on a
    handler closure, an uploaded callable or a client-local pathname.
    """

    request: bytes
    config: bytes

    def __post_init__(self) -> None:
        if any(
            type(body) is not bytes or not 0 < len(body) <= MAX_METADATA_BYTES
            for body in (self.request, self.config)
        ):
            raise ValueError("creation/configuration metadata exceeds its byte bound")

    @classmethod
    def encode(cls, creation: TrainingCreation) -> "CreationRecord":
        if not isinstance(creation, TrainingCreation):
            raise TypeError("creation requires validated portable settings")
        return cls(encode(creation), encode(creation.config))

    @property
    def reference(self) -> ArtifactRef:
        return _reference("creation", self.request)

    @property
    def configuration(self) -> ArtifactRef:
        return _reference("configuration", self.config)

    def publish(self, artifacts: ArtifactStore) -> None:
        for ref, body in (
            (self.configuration, self.config),
            (self.reference, self.request),
        ):

            def source(_: ArtifactRef, data: bytes = body) -> io.BytesIO:
                return io.BytesIO(data)

            artifacts.put(ref, source)


def load_creation(artifacts: ArtifactStore, info: RunInfo) -> TrainingCreation:
    """Read bounded immutable choices and recheck catalog/config correspondence."""
    if artifacts.owner_id != info.owner_id:
        raise PermissionError("creation storage belongs to another owner")
    reference = artifacts.get("creation." + info.request_sha256)
    if (
        reference is None
        or reference.sha256 != info.request_sha256
        or reference.media_type != CONTROL_MEDIA_TYPE
        or reference.size_bytes > MAX_METADATA_BYTES
    ):
        raise ValueError("registered creation metadata is missing or incompatible")
    with artifacts.open(reference) as stream:
        body = stream.read(MAX_METADATA_BYTES + 1)
    if (
        len(body) != reference.size_bytes
        or hashlib.sha256(body).hexdigest() != reference.sha256
    ):
        raise ValueError("registered creation metadata failed its content checksum")
    value = decode(body)
    if not isinstance(value, TrainingCreation):
        raise ValueError("registered creation metadata has the wrong record type")
    expected = CreationRecord.encode(value)
    if expected.configuration != info.configuration:
        raise ValueError("creation settings differ from the registered configuration")
    if artifacts.get(info.configuration.id) != info.configuration:
        raise ValueError("registered configuration is not in owner storage")
    with artifacts.open(info.configuration) as stream:
        if stream.read(MAX_METADATA_BYTES + 1) != expected.config:
            raise ValueError("registered configuration failed its content checksum")
    return value
