"""Scalar history survives completion races, disconnects and disposable indexes."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from engine_tests.server.test_http import OTHER_TOKEN, ROOT, TOKEN, endpoint
from engine_tests.server.test_request_records import request, result, store
from engine_tests.server.test_service_client import address
from shadowspill_client import ServiceClient
from shadowspill_client.metrics import MetricQueries
from shadowspill_client.protocol.codec import MAX_METADATA_BYTES, encode
from shadowspill_client.protocol.metrics import MetricRecord
from shadowspill_client.protocol.results import ExecutionReceipt, Failure
from shadowspill_client.transport.http import RemoteError


def finish(records, command, *, loss=1.25, metrics=None, history="history", updates=1):
    records.submit(command)
    records.accept(command.identity.request_id)
    records.dispatch(command.identity.request_id)
    completed = replace(
        result(command),
        loss=loss,
        metrics={"weight": 3} if metrics is None else metrics,
        history_id=history,
        completed_updates=updates,
    )
    records.executed(ExecutionReceipt.from_result(completed))
    records.complete(completed)
    return completed


def test_completion_cursor_retains_late_results_and_fixed_snapshot_after_reopen(
    tmp_path,
):
    first, second = request("first", 1), request("second", 2)
    with store(tmp_path) as records:
        records.submit(first)
        records.submit(second)
        records.fail("second", Failure("invalid", "invalid input", "not_started", True))
        before = records.metrics()
        assert [row.identity.request_id for row in before.records] == ["second"]
        assert (
            before.records[0].failure_code == "invalid"
            and before.records[0].loss is None
        )
        finish(records, first, loss=7.5, updates=9)
        after = records.metrics(after=before.through)
        assert [row.identity.request_id for row in after.records] == ["first"]
        assert after.records[0].identity.sequence < before.records[0].identity.sequence
        assert records.metrics(through=before.through) == before
        finish(records, request("rewound", 3), history="restored", updates=2)
        page = records.metrics(limit=1)
        assert page.next_cursor == before.records[0].position
        second_page = records.metrics(
            after=page.next_cursor, through=page.through, limit=1
        )
        assert second_page.records[0].completed_updates == 9
        end = records.metrics(after=second_page.next_cursor, through=page.through)
        assert (
            end.records[0].completed_updates == 2
            and end.records[0].history_id == "restored"
        )
        assert end.next_cursor is None
        stable = records.metrics()
    with store(tmp_path) as reopened:
        assert reopened.metrics() == stable
        assert reopened.metrics(after=stable.through).records == ()
        assert reopened.metrics(through=before.through) == before


def test_query_includes_confirmed_failure_without_inventing_loss(tmp_path):
    with store(tmp_path) as records:
        command = replace(request(), operation="forward_backward", mode="training")
        records.submit(command)
        records.accept("a")
        records.dispatch("a")
        completed = result(command)
        records.executed(ExecutionReceipt.from_result(completed))
        records.complete(completed)
        failed = request("failed", 2)
        records.submit(failed)
        records.accept("failed")
        records.dispatch("failed")
        completed = replace(result(failed), loss=1.125)
        receipt = ExecutionReceipt.from_result(completed)
        records.executed(receipt)
        records.fail(
            "failed",
            Failure(
                "artifact_failed",
                "publication failed",
                "completed",
                True,
                completed_updates=receipt.completed_updates,
                history_id=receipt.history_id,
                accumulation=receipt.accumulation,
                loss=completed.loss,
                metrics=completed.metrics,
            ),
        )
        rows = records.metrics().records
        assert rows[0].outcome == "completed" and rows[0].completed_updates == 0
        assert (
            rows[1].execution_effect == "completed"
            and rows[1].failure_code == "artifact_failed"
        )
        assert rows[1].loss == 1.125 and rows[1].metrics == completed.metrics


def test_metric_page_stops_at_codec_byte_bound_without_losing_the_next_record(tmp_path):
    with store(tmp_path) as records:
        for i in range(4):
            finish(records, request(f"large-{i}", i + 1), metrics={"m" * 400000: i})
        first = records.metrics(limit=100)
        assert len(first.records) == 2 and first.next_cursor is not None
        assert len(encode(first)) <= MAX_METADATA_BYTES
        second = records.metrics(after=first.next_cursor, through=first.through)
        assert len(second.records) == 2 and second.next_cursor is None
        assert {row.identity.sequence for row in (*first.records, *second.records)} == {
            1,
            2,
            3,
            4,
        }


@pytest.mark.parametrize(
    "query",
    [
        {"after": -1},
        {"after": True},
        {"limit": 0},
        {"limit": 129},
        {"limit": True},
        {"through": 999},
        {"through": 1, "after": 2},
        {"after": 1 << 63},
    ],
)
def test_invalid_cursor_or_count_is_rejected_without_advancing_journal(tmp_path, query):
    with store(tmp_path) as records:
        before = records.metrics()
        with pytest.raises(ValueError):
            records.metrics(**query)
        assert records.metrics() == before


def test_public_query_needs_no_workers_and_preserves_cursor_through_service_reconnect(
    tmp_path,
):
    with store(tmp_path) as records:
        expected = finish(records, request(), loss=4.5)
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
            page = service.get_metrics("run", limit=1).result(5)
            assert page.records[0].identity == expected.identity
            assert (
                page.records[0].loss == expected.loss
                and page.records[0].metrics == expected.metrics
            )
            if attempt == 0:
                retained = page
            else:
                assert page == retained
            assert (
                service.get_metrics("run", after=page.through).result(5).records == ()
            )
            with pytest.raises(RemoteError, match="not_found"):
                http.connect(OTHER_TOKEN).request("GET", ROOT + "/metrics")
            for query in (
                "after=-1",
                "through=99999",
                "limit=129",
                "bad=1",
                "after=0&after=1",
            ):
                with pytest.raises(RemoteError, match="invalid_request"):
                    http.client.request("GET", ROOT + "/metrics?" + query)
        assert service._memory.used == service._work.pending.used == (0, 0)


def test_jsonl_export_pins_its_range_while_later_results_arrive(tmp_path, monkeypatch):
    with (
        store(tmp_path) as records,
        endpoint(
            tmp_path, run=SimpleNamespace(journal=records, controller=None)
        ) as http,
        ServiceClient(
            address(http.client), token=TOKEN, session_id="session"
        ) as service,
    ):
        for i in range(3):
            finish(records, request(f"first-{i}", i + 1), loss=i + 0.25)
        expected = records.metrics()
        original = service._request
        calls = []

        def read(method, route, kind, value=None):
            result = original(method, route, kind, value)
            calls.append(route)
            if len(calls) == 1:
                finish(records, request("late", 4), loss=9)
            return result

        monkeypatch.setattr(service, "_request", read)
        metrics = MetricQueries(service, "session", "run")
        target = metrics.export_jsonl(tmp_path / "metrics.jsonl", limit=1).result(5)
        # Plain JSON objects, one per record, in the records' own field form.
        rows = [json.loads(line) for line in target.read_text().splitlines()]
        assert rows == [row.to_dict() for row in expected.records]
        assert tuple(MetricRecord.from_dict(row) for row in rows) == expected.records
        assert len(calls) == 3 and all("/metrics?" in path for path in calls)
        assert all(f"through={expected.through}" in path for path in calls[1:])
        with pytest.raises(FileExistsError):
            metrics.export_jsonl(target).result(5)
        assert len(calls) == 3 and target.is_file()
        later = metrics.query(after=expected.through).result(5)
        assert [row.identity.request_id for row in later.records] == ["late"]
        empty = metrics.export_jsonl(
            tmp_path / "empty.jsonl", after=later.through
        ).result(5)
        assert empty.read_bytes() == b""
    assert service._memory.used == service._work.pending.used == (0, 0)


def test_failed_jsonl_export_removes_its_partial_file_and_preserves_results(
    tmp_path, monkeypatch
):
    with (
        store(tmp_path) as records,
        endpoint(
            tmp_path, run=SimpleNamespace(journal=records, controller=None)
        ) as http,
        ServiceClient(
            address(http.client), token=TOKEN, session_id="session"
        ) as service,
    ):
        for i in range(3):
            finish(records, request(f"row-{i}", i + 1))
        original, calls = service._request, []

        def fail(method, route, kind, value=None):
            calls.append(route)
            if len(calls) == 2:
                raise OSError("inspection disconnected")
            return original(method, route, kind, value)

        metrics = MetricQueries(service, "session", "run")
        destination, keep = tmp_path / "partial.jsonl", tmp_path / "keep"
        keep.write_text("unrelated")
        before = records.metrics()
        with monkeypatch.context() as patch:
            patch.setattr(service, "_request", fail)
            with pytest.raises(OSError, match="disconnected"):
                metrics.export_jsonl(destination, limit=1).result(5)
        assert not destination.exists() and keep.read_text() == "unrelated"
        assert records.metrics() == before
        assert metrics.export_jsonl(destination, limit=1).result(5) == destination
    assert service._memory.used == service._work.pending.used == (0, 0)
