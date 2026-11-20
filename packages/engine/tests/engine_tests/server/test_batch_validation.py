"""Local preview and automatic submission use the server's immutable layouts."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from engine_tests.fixtures.component_operations import records
from engine_tests.server.test_training_client import runtime
from engine_tests.training.test_prepared_requests import family, variable_inventory
from shadowspill.training.preparation.artifacts.inspection import summary_rows
from shadowspill.training.session.prepared import PreparedRequests
from shadowspill_client import BatchConfig, Datum
from shadowspill_client.batch_validation import BatchValidator
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.inputs.selection import validate_snapshot
from shadowspill_client.protocol.codec import digest
from shadowspill_client.protocol.plans import PlanSummaryPage
from shadowspill_client.protocol.requests import ArtifactRef


def validator(prepared, change=lambda row: row):
    config = prepared.inventory.layouts.setup.config
    ref = ArtifactRef("plans", "a" * 64, 0)
    rows = [change(row) for row in summary_rows(prepared)]
    pages = [
        PlanSummaryPage(
            ref,
            digest(config),
            i // 32,
            i // 32 + 1 if i + 32 < len(rows) else None,
            sum(map(len, prepared.families.values())),
            sum(map(len, prepared.plans.values())),
            rows[i : i + 32],
        )
        for i in range(0, len(rows), 32)
    ]
    requests = []

    def fetch(method, path, result_type):
        requests.append(path)
        return pages[len(requests) - 1]

    return BatchValidator(SimpleNamespace(_request=fetch), config, ref), requests


@pytest.mark.parametrize("operation", ["step", "forward_backward", "forward"])
def test_preview_and_server_choose_same_candidate_and_cache_bounded_metadata(
    tmp_path, operation
):
    prepared = family()
    preview, requests = validator(prepared)
    budget = BufferBudget(preview.config.execution.max_staging_bytes, 1)
    for count in (1, 3, 4):
        batch = (records() * 2)[:count]
        if operation == "forward":
            batch = [replace(datum, loss_inputs={}) for datum in batch]
        with preview.snapshot(batch, budget, operation, None, None) as snapshot:
            result = preview.validate(snapshot, operation, None, None)
            server = PreparedRequests(prepared, cache_root=tmp_path)
            selected, assignment = server._select(
                result.operation,
                validate_snapshot(snapshot, preview.config, result.operation),
            )
            assert result.packing_checked and result.candidate == selected.id
            assert result.scheduled_capacities == assignment.schedule.capacities(
                preview.config.inputs
            )
        assert budget.used == (0, 0)
        if count == 1:
            first_requests = list(requests)
        else:
            assert requests == first_requests


def test_preview_never_uploads_and_bad_input_consumes_no_sequence(
    tmp_path, monkeypatch
):
    with runtime(tmp_path) as (service, training, run, _):
        original = service._transport.upload

        def forbidden(*args):
            pytest.fail("preview/invalid batch uploaded input")

        monkeypatch.setattr(service._transport, "upload", forbidden)
        result = training.validate_batch(records()).result(5)
        assert result.packing_checked and result.candidate
        assert (
            training._next == 1
            and run.journal.last_sequence == 0
            and not run.workers.calls
        )
        malformed = records()
        malformed[0] = replace(
            malformed[0], loss_inputs={**malformed[0].loss_inputs, "class_id": -1}
        )
        with pytest.raises(ValueError):
            training.step(malformed)
        assert training._next == 1 and run.journal.last_sequence == 0
        assert training._slots.used == training._snapshots.used == (0, 0)
        monkeypatch.setattr(service._transport, "upload", original)
        assert training.step(records()).result(5).completed_updates == 1
        training.close().result(5)


def test_custom_grouping_reports_server_dependency_without_guessing():
    inv = variable_inventory(
        [BatchConfig(limits={"x": 128, "y": 128, "records": 3})],
        groups=lambda config, batch: [[0, 1, 2]],
    )
    preview, _ = validator(family(inv))
    records = [Datum({"x": np.zeros(40), "y": np.zeros(1)}) for _ in range(3)]
    with preview.snapshot(
        records, BufferBudget(8 << 20), "forward", None, None
    ) as snapshot:
        result = preview.validate(snapshot, "forward", None, None)
        assert not result.packing_checked and result.candidate is None
        assert "groups inputs on the server" in result.note


@pytest.mark.parametrize("operation", ["step", "forward_backward"])
@pytest.mark.parametrize("empty", [False, True])
def test_zero_participation_rejects_locally_without_upload_or_state_change(
    tmp_path, monkeypatch, operation, empty
):
    with runtime(tmp_path) as (service, training, run, _):
        original = service._transport.upload
        monkeypatch.setattr(
            service._transport,
            "upload",
            lambda *args: pytest.fail("empty training batch uploaded"),
        )
        batch = [] if empty else records(masked=True)
        with pytest.raises(ValueError, match="no loss participation"):
            training.validate_batch(batch, operation=operation).result(5)
        with pytest.raises(ValueError, match="no loss participation"):
            getattr(training, operation)(batch)
        assert training._next == 1 and run.journal.last_sequence == 0
        assert not run.workers.calls
        assert training._slots.used == training._snapshots.used == (0, 0)
        monkeypatch.setattr(service._transport, "upload", original)
        # Entirely masked records can coexist with supervised records. Numeric
        # zero weights do not turn valid data into a rejected empty contribution.
        mixed = [
            records(masked=True)[0],
            replace(
                records()[1], loss_inputs={**records()[1].loss_inputs, "weights": 0.0}
            ),
        ]
        assert training.step(mixed).result(5).completed_updates == 1
        assert training.forward(records(masked=True)).result(5).completed_updates == 1
        training.close().result(5)


def test_original_batch_alternatives_are_not_combined_by_client():
    inv = variable_inventory(
        [
            BatchConfig(limits={"x": x, "y": y, "records": 2})
            for x, y in ((8, 2), (2, 8))
        ]
    )
    preview, _ = validator(family(inv))
    records = [Datum({"x": np.zeros(3), "y": np.zeros(3)})]
    with (
        preview.snapshot(
            records, BufferBudget(8 << 20), "forward", None, None
        ) as snapshot,
        pytest.raises(ValueError, match="no requested batch option"),
    ):
        preview.validate(snapshot, "forward", None, None)
