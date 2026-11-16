"""Standalone HTML from saved inspection data; no live workers or Plan loading."""

import hashlib
from collections.abc import Iterator, Mapping

from shadowspill.diagnostics.occupancy.page import html_parts, script_json
from shadowspill.orchestration.launch.publication import OutputSpec, OutputWriter
from shadowspill.orchestration.launch.staging import COPY_BYTES
from shadowspill_client.protocol.codec import MAX_METADATA_BYTES, decode
from shadowspill_client.protocol.plans import (
    PLAN_REPORT_MEDIA_TYPE,
    PlanReportRequest,
    PlanSummaryPage,
    PlanSummaryRow,
    summary_id,
)
from shadowspill_client.protocol.requests import ArtifactRef

from .artifacts import ArtifactStore
from .records.artifact_index import ArtifactAlias
from .records.requests import RunIdentity


def _rows(store: ArtifactStore, plans: ArtifactRef) -> Iterator[PlanSummaryRow]:
    page = 0
    while True:
        reference = store.get(summary_id(plans, page))
        if reference is None:
            raise ValueError("saved Plan inspection data is unavailable")
        with store.open(reference) as stream:
            result = decode(stream.read(MAX_METADATA_BYTES + 1))
        if not isinstance(result, PlanSummaryPage) or (
            result.plans != plans or result.page != page
        ):
            raise ValueError("inspection page differs from the requested Plans")
        yield from result.rows
        if result.next_page is None:
            return
        page = result.next_page


def _render(store: ArtifactStore, plans: ArtifactRef) -> Iterator[bytes]:
    prefix, suffix = html_parts("Prepared training Plans")
    yield prefix.encode()
    yield b'{"rows":['
    for index, row in enumerate(_rows(store, plans)):
        yield (b"," if index else b"") + script_json(row.to_dict()).encode()
    yield b'],"views":['
    count = 0
    for row in _rows(store, plans):
        if row.kind != "worker_plan":
            continue
        if not row.values.get("views"):
            raise ValueError("saved Plan detailed report data is unavailable")
        views = row.values["views"]
        if not isinstance(views, (tuple, list)):
            raise ValueError("invalid saved Plan views")
        for view in views:
            if not isinstance(view, Mapping):
                raise ValueError("invalid saved Plan view reference")
            data = view.get("data")
            if not isinstance(data, Mapping):
                raise ValueError("invalid saved Plan view reference")
            reference = ArtifactRef.from_dict(data)
            if store.get(reference.id) != reference:
                raise ValueError("Plan view differs from its saved reference")
            yield (b"," if count else b"") + b'{"rank":'
            yield script_json(row.values["rank"]).encode() + b',"id":'
            yield script_json(row.values["id"]).encode() + b',"mode":'
            yield script_json(view["mode"]).encode() + b',"data":'
            with store.open(reference) as stream:
                while chunk := stream.read(COPY_BYTES):
                    # Keep embedded JSON safe even when chunks split HTML tags.
                    yield chunk.replace(b"<", b"\\u003c")
            yield b"}"
            count += 1
    if not count:
        raise ValueError("saved Plans contain no report views")
    yield b"]}"
    yield suffix.encode()


def generate_plan_report(
    store: ArtifactStore, identity: RunIdentity, request: PlanReportRequest
) -> ArtifactRef:
    if store.owner_id != identity.owner_id:
        raise PermissionError("Plan report crossed authenticated owner scope")
    if (request.session_id, request.run_id) != (identity.session_id, identity.run_id):
        raise ValueError("Plan report belongs to another run")
    name = request.report_id
    alias = ArtifactAlias(identity.session_id, identity.run_id, name, name, "report")

    def saved() -> ArtifactRef | None:
        reference = store.resolve_alias(
            alias.session_id, alias.run_id, alias.name, kind="report"
        )
        if reference is not None and reference.id != name:
            raise ValueError("Plan report registration differs from its request")
        return reference

    reference = saved()
    if reference is not None:
        return reference
    spec = OutputSpec(name, request.max_bytes, PLAN_REPORT_MEDIA_TYPE, exact_size=False)

    def write(writer: OutputWriter) -> ArtifactRef:
        checksum, size = hashlib.sha256(), 0
        for chunk in _render(store, request.plans):
            view = memoryview(chunk)
            while view:
                written = writer.write(view[:COPY_BYTES])
                checksum.update(view[:written])
                size += written
                view = view[written:]
        return ArtifactRef(name, checksum.hexdigest(), size, spec.media_type)

    try:
        return store.produce_group(
            (spec,), lambda _, writer: write(writer), alias=alias
        )[0]
    except (ValueError, RuntimeError):
        reference = saved()
        if reference is None:
            raise
        return reference
