"""Caller annotations retain explicit request identity without changing execution."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from engine_tests.fixtures.definition_operations import records as inputs
from engine_tests.server.test_http import OTHER_TOKEN, ROOT, TOKEN, endpoint
from engine_tests.server.test_metrics import finish
from engine_tests.server.test_request_records import request, store
from engine_tests.server.test_service_client import address
from engine_tests.server.test_training_client import runtime
from shadowspill.server.records import Journal
from shadowspill_client import ServiceClient
from shadowspill_client.metrics import MetricQueries
from shadowspill_client.protocol.codec import decode, encode
from shadowspill_client.protocol.metrics import MetricAnnotation
from shadowspill_client.protocol.results import Failure
from shadowspill_client.transport.http import RemoteError


def test_late_caller_metrics_keep_original_history_and_a_separate_snapshot_position(
    tmp_path,
):
    annotation = MetricAnnotation("first", {"loss": 91, "eval/reward": 0.75}, "eval")
    with store(tmp_path) as records:
        first = finish(records, request("first", 1), loss=3.5, updates=9)
        before = records.metrics()
        finish(records, request("restored", 2), history="restored-history", updates=2)
        added = records.log_metrics(annotation)
        assert added.source == "caller" and added.annotation_id == "eval"
        assert added.identity == first.identity and added.history_id == "history"
        assert added.completed_updates == 9 and added.loss is None
        assert added.metrics == annotation.values
        assert records.metrics(through=before.through) == before
        assert records.get("first")[1].result == first
        assert records.last_sequence == 2
        rows = records.metrics(after=before.through, limit=1)
        assert rows.records[0].source == "engine" and rows.next_cursor is not None
        end = records.metrics(after=rows.next_cursor, through=rows.through)
        assert end.records == (added,) and end.next_cursor is None
        stable = records.metrics()
        assert records.log_metrics(annotation) == added
        assert records.metrics() == stable
        with pytest.raises(ValueError, match="different"):
            records.log_metrics(replace(annotation, values={"eval/reward": 0.9}))
        assert records.metrics() == stable
    with store(tmp_path) as reopened:
        assert reopened.metrics() == stable
        assert reopened.log_metrics(annotation) == added
        assert reopened.metrics(through=before.through) == before


@pytest.mark.parametrize("state", ["uploading", "accepted", "running", "absent"])
def test_annotations_require_a_known_terminal_request(tmp_path, state):
    with store(tmp_path) as records:
        if state != "absent":
            records.submit(request())
        if state in ("accepted", "running"):
            records.accept("a")
        if state == "running":
            records.dispatch("a")
        before = records.metrics()
        with pytest.raises((ValueError, KeyError)):
            records.log_metrics(MetricAnnotation("a", {"reward": 1}, "eval"))
        assert records.metrics() == before


def test_annotations_on_failures_never_invent_engine_effects_or_scalars(tmp_path):
    with store(tmp_path) as records:
        records.submit(request())
        records.fail("a", Failure("invalid", "rejected input", "not_started", True))
        added = records.log_metrics(MetricAnnotation("a", {"rejected": 1}, "eval"))
        assert added.source == "caller" and added.outcome == "failed"
        assert added.history_id is added.completed_updates is added.loss is None
        assert (
            added.execution_effect == "not_started" and added.failure_code == "invalid"
        )
        assert added.metrics == {"rejected": 1}
        assert not records.metrics().records[0].metrics


def test_lost_annotation_commit_reply_is_reconciled_without_a_duplicate(
    tmp_path, monkeypatch
):
    annotation = MetricAnnotation("a", {"reward": 1}, "eval")
    with store(tmp_path) as records:
        finish(records, request())
        original = records._index.add_annotation

        def fail(row):
            original(row)
            raise OSError("lost cache/commit reply")

        monkeypatch.setattr(records._index, "add_annotation", fail)
        with pytest.raises(OSError, match="lost"):
            records.log_metrics(annotation)
        with pytest.raises(RuntimeError, match="unavailable"):
            records.log_metrics(annotation)
    with store(tmp_path) as reopened:
        page = reopened.metrics()
        assert len(page.records) == 2
        assert reopened.log_metrics(annotation) == page.records[-1]
        assert reopened.metrics() == page


def test_public_log_metrics_does_not_submit_workers_or_change_training_counters(
    tmp_path,
):
    with runtime(tmp_path) as (service, training, run, _):
        completed = training.step(inputs()).result(5)
        before = training.metrics.query().result(5)
        sequence, calls = training._next, list(run.workers.calls)
        values = {"loss": 90, "reward": 0.25}
        added = training.log_metrics(values, request_id=completed.request_id).result(5)
        assert added.source == "caller" and added.loss is None
        assert added.completed_updates == completed.completed_updates
        assert (
            training.log_metrics(values, request_id=completed.request_id).result(5)
            == added
        )
        values["reward"] = 99
        assert added.metrics["reward"] == 0.25
        assert run.workers.calls == calls and training._next == sequence
        assert training.metrics.query(through=before.through).result(5) == before
        assert (
            training.requests.get(completed.request_id).result(5).record
            == completed.record
        )
        training.close().result(5)
        late = training.log_metrics(
            {"eval/reward": 1.5},
            request_id=completed.request_id,
            metric_id="external-eval",
        ).result(5)
        assert late.identity == added.identity and late.annotation_id == "external-eval"
        rows = training.metrics.query().result(5)
        path = training.metrics.export_jsonl(
            tmp_path / "metrics.jsonl", limit=1
        ).result(5)
        assert (
            tuple(decode(line) for line in path.read_bytes().splitlines()[1:])
            == rows.records
        )
    assert service._memory.used == service._work.pending.used == (0, 0)


def test_authenticated_annotation_after_restart_and_lost_http_reply(
    tmp_path, monkeypatch
):
    with store(tmp_path) as records:
        finish(records, request())
    for attempt in range(2):
        with (
            store(tmp_path) as records,
            endpoint(
                tmp_path, run=SimpleNamespace(journal=records, controller=None)
            ) as http,
            ServiceClient(
                address(http.client), token=TOKEN, session_id="session"
            ) as service,
        ):
            metrics = MetricQueries(service, "session", "run")
            original = service._request
            if attempt == 0:

                def lost(*args, _original=original, **kwargs):
                    _original(*args, **kwargs)
                    raise ConnectionError("lost HTTP reply")

                with monkeypatch.context() as patch:
                    patch.setattr(service, "_request", lost)
                    with pytest.raises(ConnectionError, match="lost"):
                        metrics._log({"reward": 2.0}, request_id="a").result(5)
            added = metrics._log({"reward": 2.0}, request_id="a").result(5)
            assert len(records.metrics().records) == 2
            assert added == records.metrics().records[-1]
            with pytest.raises(RemoteError, match="not_found"):
                http.connect(OTHER_TOKEN).request(
                    "POST",
                    ROOT + "/metrics",
                    MetricAnnotation("a", {"reward": 3}, "other"),
                )
            with pytest.raises(RemoteError, match="invalid_request"):
                http.client.request("POST", ROOT + "/metrics", added)
        assert service._memory.used == service._work.pending.used == (0, 0)


def test_annotation_codec_snapshots_only_bounded_numeric_values():
    values = {"reward": 1.5}
    annotation = MetricAnnotation("a", values, "eval")
    values["reward"] = 99
    assert annotation.values["reward"] == 1.5
    assert decode(encode(annotation)) == annotation


@pytest.mark.parametrize("change", ["history", "position", "source", "repeat"])
def test_replay_rejects_annotations_detached_from_their_original_result(
    tmp_path, change
):
    with store(tmp_path) as records:
        finish(records, request())
        row = records.log_metrics(MetricAnnotation("a", {"reward": 1}, "eval"))
    changed = replace(row, position=row.position + 1, annotation_id="another")
    if change == "history":
        changed = replace(changed, history_id="unrelated")
    elif change == "position":
        changed = replace(changed, position=row.position)
    elif change == "source":
        changed = replace(changed, source="engine", annotation_id=None)
    else:
        changed = replace(changed, annotation_id=row.annotation_id)
    with Journal(tmp_path / "journal") as journal:
        journal.append([("metrics.caller", changed.to_dict())])
    with pytest.raises(ValueError, match="caller journal"):
        store(tmp_path)


def test_caller_history_pages_obey_codec_limits_and_keep_every_annotation(tmp_path):
    values = {f"metric-{i}": i for i in range(128)}
    with store(tmp_path) as records:
        finish(records, request())
        before = records.metrics()
        expected = [
            records.log_metrics(MetricAnnotation("a", values, f"eval-{i}"))
            for i in range(70)
        ]
        page = records.metrics(after=before.through, limit=128)
        assert len(page.records) < len(expected) and page.next_cursor is not None
        found = list(page.records)
        while page.next_cursor is not None:
            page = records.metrics(
                after=page.next_cursor, through=page.through, limit=128
            )
            found.extend(page.records)
        assert found == expected


@pytest.mark.parametrize(
    "values",
    [
        {},
        {"bad": True},
        {"bad": float("nan")},
        {"bad": []},
        {"": 1},
        {" bad": 1},
        {"x" * 257: 1},
        {str(i): i for i in range(129)},
    ],
)
def test_invalid_or_oversized_caller_metrics_are_rejected_before_submission(values):
    with pytest.raises((ValueError, TypeError)):
        MetricAnnotation("a", values, "eval")
