"""Bounded reports consume durable outcomes, never execution or tensor payloads."""

import hashlib
import io
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from engine_tests.server.test_artifacts import store as artifact_store
from engine_tests.server.test_http import OTHER_TOKEN, ROOT, TOKEN, endpoint
from engine_tests.server.test_metrics import finish
from engine_tests.server.test_request_records import request, store
from engine_tests.server.test_service_client import address
from engine_tests.server.test_training_client import runtime
from shadowspill.server.reports import _render, generate_report
from shadowspill_client import ServiceClient
from shadowspill_client.protocol.codec import decode, encode
from shadowspill_client.protocol.metrics import MetricAnnotation, MetricSnapshot
from shadowspill_client.protocol.reports import ReportSettings
from shadowspill_client.protocol.requests import ArtifactRef
from shadowspill_client.reports import report
from shadowspill_client.transport.http import RemoteError


@pytest.mark.parametrize("format", ["html", "json"])
def test_report_preserves_values_scope_and_snapshot_after_restart(tmp_path, format):
    with store(tmp_path) as records:
        finish(records, request(), loss=3.75, metrics={"<script>alert(1)</script>": 2})
        records.log_metrics(MetricAnnotation("a", {"reward": 0.5}, "eval"))
        before = records.metrics()
    with (
        store(tmp_path) as records,
        endpoint(
            tmp_path, run=SimpleNamespace(journal=records, controller=None)
        ) as http,
    ):
        with ServiceClient(
            address(http.client), token=TOKEN, session_id="session"
        ) as service:
            handle = report(
                service, "session", "run", format=format, max_bytes=1 << 20
            ).result(5)
            assert decode(encode(handle.document)) == handle.document
            path = handle.download(tmp_path / ("report." + format)).result(5)
            if format == "json":
                data = json.loads(path.read_bytes())
                assert data["records"] == [row.to_dict() for row in before.records]
                assert data["record_counts"] == {"engine": 1, "caller": 1}
            else:
                body = path.read_text()
                assert "<script>" not in body and "&lt;script&gt;" in body
                assert "3.75" in body and "reward" in body
                assert "1 engine records; 1 caller annotations" in body
            assert records.metrics() == before
            finish(records, request("later", 2), loss=8)
            saved = report(
                service,
                "session",
                "run",
                format=format,
                through=before.through,
                max_bytes=1 << 20,
            ).result(5)
            assert saved.document == handle.document
            with pytest.raises(RemoteError, match="not_found"):
                http.connect(OTHER_TOKEN).request(
                    "POST", ROOT + "/report", ReportSettings()
                )
        assert service._memory.used == service._work.pending.used == (0, 0)


@pytest.mark.parametrize("limit", ["records", "bytes"])
def test_report_overflow_reclaims_partial_artifacts_and_preserves_training(
    tmp_path, limit
):
    with store(tmp_path) as records, artifact_store(tmp_path) as artifacts:
        finish(records, request())
        finish(records, request("second", 2))
        before, used = records.metrics(), artifacts.used
        settings = (
            ReportSettings(max_records=1, max_bytes=1 << 20)
            if limit == "records"
            else ReportSettings(max_bytes=1)
        )
        with pytest.raises(ValueError, match=r"limit|bound"):
            generate_report(records, artifacts, settings)
        assert records.metrics() == before and artifacts.used == used
        assert artifacts.page() == ()
        document = generate_report(
            records, artifacts, ReportSettings(format="json", max_bytes=1 << 20)
        )
        assert document.reference.size_bytes < document.settings.max_bytes


def test_report_pins_before_render_and_preserves_plan_provenance(tmp_path, monkeypatch):
    with store(tmp_path) as records, artifact_store(tmp_path) as artifacts:
        finish(records, request())
        before = records.metrics()
        original = artifacts.produce_group

        def late(specs, write, *, alias):
            finish(records, request("late", 2))
            return original(specs, write, alias=alias)

        monkeypatch.setattr(artifacts, "produce_group", late)
        plans = ArtifactRef("plans", "a" * 64, 10)
        document = generate_report(
            records,
            artifacts,
            ReportSettings(format="json", max_bytes=1 << 20),
            plans=plans,
        )
        with artifacts.open(document.reference) as stream:
            data = json.load(stream)
        assert data["records"] == [row.to_dict() for row in before.records]
        assert data["plans"] == plans.to_dict()
        assert document.snapshot.through == before.through
        assert len(records.metrics().records) == 2


def test_public_training_report_after_close_uses_no_worker_or_training_sequence(
    tmp_path,
):
    with runtime(tmp_path) as (_service, training, run, _):
        training.close().result(5)
        sequence, calls = training._next, list(run.workers.calls)
        handle = training.report(format="json", max_bytes=1 << 20).result(5)
        saved = handle.download(tmp_path / "closed.json").result(5)
        data = json.loads(saved.read_bytes())
        assert data["records"][0]["operation"] == "close"
        assert calls == run.workers.calls and training._next == sequence


def test_empty_report_and_large_html_scalars_stay_bounded(tmp_path):
    with store(tmp_path) as records, artifact_store(tmp_path) as artifacts:
        empty = generate_report(
            records, artifacts, ReportSettings(format="json", max_bytes=1 << 20)
        )
        with artifacts.open(empty.reference) as stream:
            assert json.load(stream)["records"] == []
        finish(records, request(), metrics={"<" * 200000: 1})
        page = records.metrics()
        snapshot = MetricSnapshot("session", "run", 0, page.through)
        settings = ReportSettings(through=page.through)
        largest, total = 0, 0
        for fragment in _render(records, settings, snapshot, None):
            largest, total = max(largest, len(fragment)), total + len(fragment)
        assert total > 800000 and largest <= 100000


def test_report_write_failure_reclaims_its_admission_and_can_be_explicitly_retried(
    tmp_path, monkeypatch
):
    from shadowspill.orchestration.launch.publication import OutputWriter

    with store(tmp_path) as records, artifact_store(tmp_path) as artifacts:
        finish(records, request())
        before, used = records.metrics(), artifacts.used
        original, calls = OutputWriter.write, []

        def fail(writer, data):
            calls.append(1)
            if len(calls) == 3:
                raise OSError("report write failed")
            return original(writer, data)

        settings = ReportSettings(max_bytes=1 << 20)
        with monkeypatch.context() as patch:
            patch.setattr(OutputWriter, "write", fail)
            with pytest.raises(OSError, match="report write failed"):
                generate_report(records, artifacts, settings)
        assert records.metrics() == before and artifacts.used == used
        assert artifacts._disk.used == (0, 0)
        assert generate_report(records, artifacts, settings).reference.size_bytes > 0


def test_uploaded_bytes_cannot_impersonate_a_generated_report(tmp_path):
    settings = ReportSettings(format="json", max_bytes=1 << 20)
    with store(tmp_path) as records, artifact_store(tmp_path) as artifacts:
        finish(records, request())
        document = generate_report(records, artifacts, settings)
        other = tmp_path / "fresh-storage"
        other.mkdir()
        with artifact_store(other) as fresh:
            forged = replace(
                document.reference,
                size_bytes=5,
                sha256=hashlib.sha256(b"wrong").hexdigest(),
            )
            fresh.put(forged, lambda _: io.BytesIO(b"wrong"))
            with pytest.raises(ValueError, match="already committed"):
                generate_report(records, fresh, settings)
        before = artifacts.used
        assert generate_report(records, artifacts, settings) == document
        assert artifacts.used == before


def test_registered_report_survives_artifact_index_rebuild_without_rendering_again(
    tmp_path, monkeypatch
):
    from shadowspill.server import reports

    settings = ReportSettings(format="json", max_bytes=1 << 20)
    with store(tmp_path) as records, artifact_store(tmp_path) as artifacts:
        finish(records, request())
        document = generate_report(records, artifacts, settings)
    with store(tmp_path) as records, artifact_store(tmp_path) as artifacts:

        def forbidden(*_):
            pytest.fail("an already committed snapshot was rendered again")

        monkeypatch.setattr(reports, "_render", forbidden)
        assert generate_report(records, artifacts, settings) == document


@pytest.mark.parametrize(
    "changes",
    [
        {"format": "pdf"},
        {"after": -1},
        {"through": 0, "after": 1},
        {"max_records": 0},
        {"max_records": 100001},
        {"max_bytes": 0},
        {"max_bytes": 1 << 31},
    ],
)
def test_invalid_report_settings_are_rejected(changes):
    with pytest.raises(ValueError):
        replace(ReportSettings(), **changes)
