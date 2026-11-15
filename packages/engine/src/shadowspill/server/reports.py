"""Render recorded evidence with one bounded page; never run a model or a Plan."""

import hashlib
import html
import json
from collections.abc import Iterator
from dataclasses import replace

from shadowspill.orchestration.launch.publication import OutputSpec, OutputWriter
from shadowspill.orchestration.launch.staging import COPY_BYTES
from shadowspill_client.protocol.codec import encode
from shadowspill_client.protocol.metrics import MetricRecord, MetricSnapshot
from shadowspill_client.protocol.reports import (
    REPORT_MEDIA,
    REPORT_VERSION,
    ReportDocument,
    ReportSettings,
)
from shadowspill_client.protocol.requests import ArtifactRef

from .artifacts import ArtifactStore
from .records.artifact_index import ArtifactAlias
from .records.requests import RequestRecords


def _json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _rows(records: RequestRecords, settings: ReportSettings) -> Iterator[MetricRecord]:
    cursor, count = settings.after, 0
    while True:
        page = records.metrics(after=cursor, through=settings.through, limit=32)
        for row in page.records:
            count += 1
            if count > settings.max_records:
                raise ValueError(
                    "report exceeds its record limit; choose a smaller range "
                    "or explicit limit"
                )
            yield row
        if page.next_cursor is None:
            return
        cursor = page.next_cursor
        del page


def _render(
    records: RequestRecords,
    settings: ReportSettings,
    snapshot: MetricSnapshot,
    plans: ArtifactRef | None,
) -> Iterator[bytes]:
    context = {
        "format": "shadowspill.training-report",
        "version": REPORT_VERSION,
        "snapshot": snapshot.to_dict(),
        "settings": settings.to_dict(),
        "plans": None if plans is None else plans.to_dict(),
    }
    counts = {"engine": 0, "caller": 0}
    if settings.format == "json":
        yield _json(context)[:-1] + b',"records":['
    else:
        yield (
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            "<title>Training results</title><style>"
            "body{font:16px system-ui,sans-serif;max-width:1400px;margin:2rem auto;"
            "padding:0 1rem;color:#18232d}.scroll{overflow-x:auto}"
            "table{border-collapse:collapse;width:100%;min-width:850px}"
            "th,td{border-bottom:1px solid #ccd5dd;padding:.6rem;"
            "text-align:left;vertical-align:top}th{background:#edf2f6}"
            "code{overflow-wrap:anywhere}pre{white-space:pre-wrap;"
            "overflow-wrap:break-word;word-break:break-all}"
            "@media print{body{font-size:10px;margin:0;padding:0;max-width:100%}"
            "table{min-width:0;table-layout:fixed}th,td{padding:.3rem}}"
            "</style></head><body><h1>Training results</h1>"
            "<p>Recorded results; no training was rerun.</p>"
            f"<p>Session <code>{html.escape(snapshot.session_id)}</code> · "
            f"Run <code>{html.escape(snapshot.run_id)}</code></p>"
            f"<p>Journal range ({snapshot.after}, {snapshot.through}]. "
            f"Renderer version {REPORT_VERSION}.</p>"
            "<details><summary>Snapshot and Plan references</summary>"
            f"<pre>{html.escape(json.dumps(context, indent=2))}</pre></details>"
            "<p>Engine and caller values remain separate. Loss and metric values "
            "are recorded without further normalization. Update numbers can "
            "rewind after restore; histories identify the corresponding state.</p>"
            '<div class="scroll">'
            "<table><thead><tr><th>Position / request</th><th>Operation / outcome</th>"
            "<th>History / updates</th><th>Source</th><th>Loss</th>"
            "<th>Metrics</th></tr></thead><tbody>"
        ).encode()
    for index, row in enumerate(_rows(records, settings)):
        counts[row.source] += 1
        if settings.format == "json":
            yield (b"," if index else b"") + _json(row.to_dict())
        else:
            updates = row.completed_updates
            fields = [
                f"{row.position}\n{row.identity.request_id}",
                f"{row.operation}\n{row.outcome}"
                + ("\n" + row.failure_code if row.failure_code else ""),
                f"{row.history_id or 'unavailable'}\n"
                f"{updates if updates is not None else 'unavailable'}",
                row.source + ("\n" + row.annotation_id if row.annotation_id else ""),
                "—" if row.loss is None else str(row.loss),
                _json(dict(row.metrics)).decode(),
            ]
            yield b"<tr>"
            for value in fields:
                yield b"<td><pre>"
                # Escaping can expand a scalar name sixfold. Bound each escaped
                # fragment instead of building a second full rendered row.
                for start in range(0, len(value), 16 << 10):
                    yield html.escape(value[start : start + (16 << 10)]).encode()
                yield b"</pre></td>"
            yield b"</tr>"
    if settings.format == "json":
        yield b'],"record_counts":' + _json(counts) + b"}\n"
    else:
        yield (
            f"</tbody></table></div><p>{counts['engine']} engine records; "
            f"{counts['caller']} caller annotations.</p></body></html>\n"
        ).encode()


def generate_report(
    records: RequestRecords,
    store: ArtifactStore,
    settings: ReportSettings,
    *,
    plans: ArtifactRef | None = None,
) -> ReportDocument:
    """Publish one immutable snapshot through the ordinary artifact file owner.

    Called inside admitted metadata work (the HTTP scratch envelope). The
    existing output owner reserves the requested byte ceiling before rendering;
    page and escaped-row buffers are bounded independently of history length.
    """
    if store.owner_id != records.identity.owner_id:
        raise PermissionError("report publication crossed authenticated owner scope")
    first = records.metrics(after=settings.after, through=settings.through, limit=1)
    settings = replace(settings, through=first.through)
    snapshot = MetricSnapshot(
        records.identity.session_id,
        records.identity.run_id,
        settings.after,
        first.through,
    )
    del first
    identity = hashlib.sha256(
        encode(settings)
        + encode(snapshot)
        + _json(None if plans is None else plans.to_dict())
        + str(REPORT_VERSION).encode()
    ).hexdigest()
    name = "report." + identity
    alias = ArtifactAlias(
        snapshot.session_id, snapshot.run_id, identity, name, "report"
    )

    def saved() -> ArtifactRef | None:
        reference = store.resolve_alias(
            alias.session_id, alias.run_id, alias.name, kind="report"
        )
        if reference is not None and reference.id != name:
            raise ValueError("report registration differs from its snapshot")
        return reference

    # Only an atomically registered report is reusable. An arbitrary uploaded
    # file with the same predictable ID must never become a generated summary.
    reference = saved()
    if reference is None:
        spec = OutputSpec(
            name, settings.max_bytes, REPORT_MEDIA[settings.format], exact_size=False
        )

        def write(writer: OutputWriter) -> ArtifactRef:
            digest, size = hashlib.sha256(), 0
            for chunk in _render(records, settings, snapshot, plans):
                view = memoryview(chunk)
                while view:
                    written = writer.write(view[:COPY_BYTES])
                    digest.update(view[:written])
                    size += written
                    view = view[written:]
            return ArtifactRef(name, digest.hexdigest(), size, spec.media_type)

        try:
            reference = store.produce_group(
                (spec,), lambda _, writer: write(writer), alias=alias
            )[0]
        except (ValueError, RuntimeError):
            # Another explicit request may have committed this same snapshot.
            # An in-flight or failed producer stays an error; no wait/retry loop.
            reference = saved()
            if reference is None:
                raise
    return ReportDocument(snapshot, settings, reference, plans)
