"""Persist collector evidence with the existing artifact journal and aliases."""

import hashlib
from dataclasses import replace

from shadowspill.orchestration.capture.files import CaptureFiles
from shadowspill.orchestration.launch.publication import OutputSpec, OutputWriter
from shadowspill_client.protocol.captures import (
    WINDOW_BYTES,
    NsightManifest,
    NsightWindow,
    rank_name,
)
from shadowspill_client.protocol.codec import decode, encode
from shadowspill_client.protocol.requests import ArtifactRef
from shadowspill_client.protocol.service import CONTROL_MEDIA_TYPE

from .artifacts import ArtifactStore
from .records.artifact_index import ArtifactAlias
from .records.requests import RunIdentity


def read_manifest(
    store: ArtifactStore, identity: RunIdentity, rank: int
) -> NsightManifest | None:
    if store.owner_id != identity.owner_id:
        raise PermissionError("capture lookup crossed owner scope")
    reference = store.resolve_alias(
        identity.session_id, identity.run_id, rank_name(rank), kind="nsys"
    )
    if reference is None:
        return None
    if (
        reference.size_bytes > WINDOW_BYTES
        or reference.media_type != CONTROL_MEDIA_TYPE
    ):
        raise ValueError("capture manifest exceeds its metadata envelope")
    with store.open(reference) as stream:
        raw = stream.read(WINDOW_BYTES + 1)
    if hashlib.sha256(raw).hexdigest() != reference.sha256:
        raise ValueError("capture manifest failed its checksum")
    value = decode(raw, max_bytes=WINDOW_BYTES)
    if not isinstance(value, NsightManifest) or (
        value.session_id,
        value.run_id,
        value.rank,
    ) != (identity.session_id, identity.run_id, rank):
        raise ValueError("capture manifest differs from its run or rank")
    if store.get(value.plans.id) != value.plans or (
        value.report is not None and store.get(value.report.id) != value.report
    ):
        raise ValueError("capture manifest refers to unpublished artifacts")
    return value


def publish_manifest(store: ArtifactStore, value: NsightManifest) -> None:
    raw = encode(value, max_bytes=WINDOW_BYTES)
    digest = hashlib.sha256(raw).hexdigest()
    reference = ArtifactRef(
        "nsys-manifest." + digest, digest, len(raw), CONTROL_MEDIA_TYPE
    )
    alias = ArtifactAlias(
        value.session_id, value.run_id, rank_name(value.rank), reference.id, "nsys"
    )
    existing = store.get(reference.id)
    if existing is not None:
        if existing != reference:
            raise ValueError("capture manifest ID names different content")
        store.name(reference, alias)
        return

    def write(_: OutputSpec, writer: OutputWriter) -> ArtifactRef:
        part = memoryview(raw)
        while part:
            part = part[writer.write(part) :]
        return reference

    # Publish the immutable record and current-name update in one durable commit.
    store.produce_group(
        (OutputSpec(reference.id, len(raw), reference.media_type),), write, alias=alias
    )


def missing_manifest(value: NsightManifest, error: str) -> NsightManifest:
    return replace(
        value,
        status="missing",
        report=None,
        error=error,
        window=NsightWindow(
            value.window.start_step, value.window.steps, "missing", error=error
        ),
    )


def finalize_manifest(
    store: ArtifactStore, value: NsightManifest, capture: CaptureFiles | None
) -> None:
    if capture is None:
        result = missing_manifest(value, "Collector did not reach worker launch.")
    else:
        outcome = capture.outcome()  # Requires confirmed producer exit.
        result = replace(
            value,
            window=outcome.window,
            status=outcome.status,
            report=outcome.report,
            error=outcome.error,
        )
        if result.report is not None:
            try:
                store.put(result.report, capture.open)
            except Exception as error:
                # Capture delivery cannot replay or invalidate completed training.
                # If the journal is uncertain, publishing the failure below also
                # fails and retains owners for explicit reconciliation.
                message = (str(error).strip() or type(error).__name__)[:1024].rstrip()
                result = replace(result, status="failed", report=None, error=message)
    publish_manifest(store, result)
