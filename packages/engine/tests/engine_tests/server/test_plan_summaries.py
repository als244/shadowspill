"""Saved summaries use bounded durable artifacts, never live Plan loading."""

from dataclasses import replace

import pytest

from engine_tests.server.test_artifacts import reference, store
from engine_tests.server.test_http import OTHER_TOKEN, TOKEN
from engine_tests.server.test_http_creation import endpoint
from engine_tests.server.test_jobs import until
from engine_tests.server.test_local_creation import configured
from engine_tests.server.test_service_client import address
from shadowspill.server.artifacts import ArtifactStore
from shadowspill.server.plan_summaries import _publish_pages
from shadowspill_client import ServiceClient
from shadowspill_client.plans import PlansHandle
from shadowspill_client.protocol.codec import (
    MAX_METADATA_BYTES,
    ProtocolError,
    decode,
    digest,
)
from shadowspill_client.protocol.plans import PlanSummaryRow, summary_id
from shadowspill_client.transport.http import RemoteError


def pages(artifacts, plans):
    page = 0
    while True:
        ref = artifacts.get(summary_id(plans, page))
        assert ref.size_bytes <= MAX_METADATA_BYTES
        with artifacts.open(ref) as stream:
            value = decode(stream.read())
        yield value
        if value.next_page is None:
            break
        page = value.next_page


@pytest.mark.parametrize("payload", ["short", "x" * 40000, tuple(range(1024))])
def test_count_byte_and_json_node_bounds_preserve_all_rows_and_survive_reopen(
    tmp_path, payload
):
    plans = reference()
    rows = [
        PlanSummaryRow("worker_plan", {"id": str(i), "value": payload})
        for i in range(70)
    ]
    with store(tmp_path) as artifacts:
        _publish_pages(
            artifacts,
            plans,
            iter(rows),
            configuration_sha256="a" * 64,
            candidate_count=1,
            plan_count=70,
        )
        saved = list(pages(artifacts, plans))
        assert len(saved) >= 3
        assert [row for page in saved for row in page.rows] == rows
        before = artifacts.used
        _publish_pages(
            artifacts,
            plans,
            iter(rows),
            configuration_sha256="a" * 64,
            candidate_count=1,
            plan_count=70,
        )
        assert artifacts.used == before
    with store(tmp_path) as reopened:
        assert list(pages(reopened, plans)) == saved


def test_oversized_individual_row_fails_before_publication(tmp_path):
    with store(tmp_path) as artifacts:
        row = PlanSummaryRow("worker_plan", {"too_big": "x" * MAX_METADATA_BYTES})
        with pytest.raises(ProtocolError, match="exceeds"):
            _publish_pages(
                artifacts,
                reference(),
                [row],
                configuration_sha256="a" * 64,
                candidate_count=1,
                plan_count=1,
            )
        assert artifacts.used == (0, 0)


def test_public_pages_are_saved_scoped_and_do_not_read_the_plan_bundle(
    tmp_path, monkeypatch
):
    with configured(tmp_path, monkeypatch) as ctx, endpoint(ctx) as (server, _):
        with ServiceClient(address(server), token=TOKEN) as service:
            training = service.create_training_client(config=ctx.config).result(10)
            handle = training.plans
            assert isinstance(handle, PlansHandle)
            training.close().result(5)
            original = ArtifactStore._open

            def only_metadata(self, reference):
                assert reference != handle.reference, "inspection opened full Plans"
                return original(self, reference)

            monkeypatch.setattr(ArtifactStore, "_open", only_metadata)
            results, page = [], 0
            while True:
                until(lambda: ctx.http_memory.used == (0, 0))
                result = handle.summary(page=page).result(5)
                assert result.configuration_sha256 == digest(ctx.config)
                results.extend(result.rows)
                if result.next_page is None:
                    break
                page = result.next_page
            assert {row.kind for row in results} == {
                "configuration",
                "batch",
                "build_case",
                "representative",
                "candidate",
                "worker_plan",
                "coverage",
                "shared_memory",
            }
            assert (
                len([row for row in results if row.kind == "worker_plan"])
                == result.plan_count
            )
            assert all(
                row.values["simulated"] for row in results if row.kind == "worker_plan"
            )
            assert len(ctx.owners) == 1 and len(ctx.phases) == 4
            with pytest.raises(ValueError, match="integer"):
                handle.summary(page=-1)
            until(lambda: ctx.http_memory.used == (0, 0))
            with pytest.raises(RemoteError) as missing:
                handle.summary(page=999).result(5)
            assert missing.value.status == 404
        until(lambda: ctx.service._artifacts.used[1] == 0)
        until(lambda: ctx.http_memory.used == (0, 0))
        with ServiceClient(address(server), token=OTHER_TOKEN) as other:
            with pytest.raises(RemoteError) as denied:
                PlansHandle(handle.reference, other).summary().result(5)
            assert denied.value.status == 404


def test_reference_identity_prevents_same_bytes_under_different_names_colliding():
    plans = reference()
    assert summary_id(plans, 0) != summary_id(replace(plans, id="another"), 0)
    assert summary_id(plans, 0) != summary_id(plans, 1)
