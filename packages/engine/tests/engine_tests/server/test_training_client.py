"""Public runtime calls against real HTTP, journal, validation and operation state."""

from contextlib import contextmanager
from dataclasses import replace
from threading import Event, Thread

import pytest

from engine_tests.fixtures.definition_operations import records
from engine_tests.server.test_http import ROOT, TOKEN, controller_lifecycle, endpoint
from engine_tests.server.test_http_creation import endpoint as creation_endpoint
from engine_tests.server.test_jobs import until
from engine_tests.server.test_local_creation import configured as creation_service
from engine_tests.server.test_service_client import address
from engine_tests.training.test_controller import configured
from shadowspill.server.plan_summaries import _publish_pages
from shadowspill.training.preparation.artifacts.inspection import summary_rows
from shadowspill_client import ServiceClient, TrainingClient
from shadowspill_client._async import ConnectionClosed
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.creation import CreationError
from shadowspill_client.protocol.codec import digest
from shadowspill_client.protocol.payload import snapshot_batch
from shadowspill_client.protocol.requests import ArtifactRef, WriterLease
from shadowspill_client.protocol.runs import RunInfo
from shadowspill_client.results import TrainingRequestError
from shadowspill_client.transport.http import TransportError


@contextmanager
def runtime(tmp_path, **options):
    with configured(tmp_path, **options) as run, endpoint(tmp_path, run=run) as ctx:
        with ServiceClient(
            address(ctx.client), token=TOKEN, session_id="session"
        ) as service:
            reference = ArtifactRef("cpu-fixture", "a" * 64, 0)
            prepared = run.controller._prepared.prepared
            _publish_pages(
                ctx.artifacts,
                reference,
                summary_rows(prepared),
                configuration_sha256=digest(prepared.inventory.layouts.setup.config),
                candidate_count=sum(map(len, prepared.families.values())),
                plan_count=sum(map(len, prepared.plans.values())),
            )
            info = RunInfo(
                "owner",
                "session",
                "run",
                "server",
                "a" * 64,
                reference,
                state="ready",
                stage=None,
                plans=reference,
                initial_history_id="history",
            )
            # This fixture owns a controller directly, without creation jobs.
            # Its worker close is its entire retirement boundary. Real service
            # lifetime/creation envelopes are exercised separately below.
            ctx.service.jobs = controller_lifecycle(info, lambda: run.workers.closed)
            writer = service._request("POST", ROOT + "/writers/client", WriterLease)
            training = TrainingClient(
                service, info, run.controller._prepared.setup.config, writer
            )
            yield service, training, run, ctx
        assert training._slots.used == training._snapshots.used == (0, 0)
        assert service._memory.used == service._work.pending.used == (0, 0)


@pytest.mark.parametrize("disconnect", [False, True])
def test_close_future_waits_for_full_retirement_before_immediate_recreation(
    tmp_path, monkeypatch, disconnect
):
    entered, release = Event(), Event()
    calls = []

    def fault(owner):
        calls.append(owner)
        if len(calls) != 1:
            return
        original = owner.close

        def close():
            original()
            entered.set()
            assert release.wait(10)

        monkeypatch.setattr(owner, "close", close)

    with (
        creation_service(tmp_path, monkeypatch, fault=fault) as ctx,
        creation_endpoint(ctx) as (transport, _),
        ServiceClient(address(transport), token=TOKEN) as service,
    ):
        training = service.create_training_client(config=ctx.config).result(5)
        closing = training.close()
        acceptance = closing.accepted().result(5)
        try:
            assert entered.wait(5)
            reply = service._transport.request("GET", training._route)
            assert reply.value.state == "closed" and reply.status == 202
            assert ctx.jobs.used == (1, 0)
            assert training.requests.status(closing.request_id).result(5).result
            with pytest.raises(TimeoutError):
                closing.result(0.05)
            assert not closing.done() and not closing.cancel()
            assert closing.accepted().result() == acceptance
            if disconnect:
                service.close()
                with pytest.raises(ConnectionClosed):
                    closing.result(5)
                assert closing.accepted().result() == acceptance
        finally:
            release.set()
        if disconnect:
            until(lambda: ctx.jobs.used == (0, 0))
            reply = transport.request("GET", training._route)
            assert reply.status == 200 and reply.value.state == "closed"
            assert len(calls) == 1
            return
        assert closing.result(5).operation == "close"
        assert ctx.jobs.used == (0, 0)
        reply = service._transport.request("GET", training._route)
        assert reply.value.state == "closed" and reply.status == 200
        # No sleeps, capacity polling or mutation retries at the call site.
        next_run = service.create_training_client(config=ctx.config).result(5)
        assert next_run.run_id != training.run_id and len(calls) == 2
        next_run.close().result(5)
        assert ctx.jobs.used == (0, 0)


@pytest.mark.parametrize("raw", [False, True])
def test_public_creation_reattachment_fencing_and_inspection_after_ordered_close(
    tmp_path, monkeypatch, raw
):
    with (
        creation_service(tmp_path, monkeypatch) as ctx,
        creation_endpoint(ctx) as (transport, _),
        ServiceClient(address(transport), token=TOKEN) as service,
    ):
        from engine_tests.training.test_setup_resolution import arguments

        setup = dict(config=ctx.config)
        if raw:
            setup = {
                **arguments(),
                "operations": {
                    "step": ["metrics"],
                    "forward_backward": ["metrics"],
                    "forward": ["predictions"],
                },
            }
        pending = service.create_training_client(**setup, name="public client")
        assert pending.run_id and pending.session_id == service.session_id
        training = pending.result(10)
        assert training.get_config() == ctx.config and training.plans is not None
        readonly = service.get_training_client(training.run_id, read_only=True).result(
            5
        )
        with pytest.raises(PermissionError, match="read-only"):
            readonly.close()
        replacement = service.get_training_client(training.run_id).result(5)
        with pytest.raises(Exception) as fenced:
            training.close().result(5)
        assert (
            "fenc" in str(fenced.value).lower() or "writer" in str(fenced.value).lower()
        )
        closed = replacement.close()
        assert closed.accepted().result(5).identity == closed.identity
        assert closed.result(5).completed_updates == 0
        assert (
            replacement.requests.get(closed.request_id).result(5).record
            == closed.result().record
        )
        inspected = service.get_training_client(training.run_id, read_only=True).result(
            5
        )
        assert inspected.get_config() == ctx.config
        with pytest.raises(CreationError):
            service.get_training_client(training.run_id).result(5)


def test_input_and_control_snapshots_ordered_accumulation_and_explicit_recovery(
    tmp_path, monkeypatch
):
    with runtime(tmp_path) as (service, training, run, _):
        entered, release = Event(), Event()
        original = service._transport.upload

        def delayed(*args):
            entered.set()
            assert release.wait(5)
            return original(*args)

        monkeypatch.setattr(service._transport, "upload", delayed)
        values = records()
        with snapshot_batch(values, BufferBudget(8 << 20)) as expected:
            expected_digest = expected.sha256
        parameters = {"temperature": 2.0}
        first = training.forward_backward(values, loss_params=parameters)
        assert first.request_id and entered.wait(5)
        parameters["temperature"] = 999
        values[0].inputs["pixels"].fill(99)
        release.set()
        assert first.accepted().result(5).identity == first.identity
        result = first.result(5)
        assert result.accumulation.contributions == (first.request_id,)
        command = run.journal.get(first.request_id)[0]
        assert command.payload.sha256 == expected_digest
        assert command.loss_params == {"temperature": 2.0}
        assert result.loss_params == {"temperature": 2.0}

        entered.clear()
        release.clear()
        malformed = records()
        malformed[0] = replace(
            malformed[0], loss_inputs={**malformed[0].loss_inputs, "class_id": -1}
        )
        sequence = training._next
        with pytest.raises(ValueError):
            training.forward_backward(malformed)
        assert training._next == sequence and not entered.is_set()
        bad = training.forward_backward(records(), loss_params={"temperature": -1})
        assert entered.wait(5)
        dependent = training.optim_step()
        release.set()
        with pytest.raises(TrainingRequestError) as error:
            bad.result(5)
        assert error.value.failure.execution_effect == "not_started"
        with pytest.raises(TrainingRequestError) as suffix:
            dependent.result(5)
        assert bad.request_id in suffix.value.failure.failed_dependencies
        assert run.workers.calls == [first.request_id]
        corrected = training.forward_backward(records()).result(5)
        assert len(corrected.accumulation.contributions) == 2
        updated = training.optim_step(optim={"lr": 0.03}, grad_scale=0.5).result(5)
        assert updated.completed_updates == 1 and not updated.accumulation.open
        assert updated.optim["lr"] == pytest.approx(0.03)
        assert updated.grad_scale == 0.5
        assert training.close().result(5).completed_updates == 1


def test_disconnect_after_acceptance_preserves_server_execution_without_replay(
    tmp_path,
):
    with runtime(tmp_path, blocked=True) as (service, training, run, ctx):
        pending = training.step(records())
        pending.accepted().result(5)
        assert run.workers.entered.wait(5)
        service.close()
        with pytest.raises(ConnectionClosed):
            pending.result(5)
        assert not run.workers.closed and run.workers.calls == [pending.request_id]
        run.workers.release.set()
        with ServiceClient(
            address(ctx.client), token=TOKEN, session_id="session"
        ) as reconnect:
            status = reconnect._request(
                "GET",
                ROOT + "/requests/" + pending.request_id + "?wait=completed&timeout=1",
                type(run.controller.status(pending.request_id)),
            )
            assert status.result.completed_updates == 1
            assert run.workers.calls == [pending.request_id]


def test_lost_submission_reply_blocks_unsent_suffix_and_keeps_stable_ids(
    tmp_path, monkeypatch
):
    with runtime(tmp_path) as (service, training, run, _):
        training.validate_batch(records()).result(5)  # Cache immutable metadata.
        entered, release = Event(), Event()

        def lost(*args, **kwargs):
            entered.set()
            assert release.wait(5)
            raise TransportError("fixture lost registration reply")

        monkeypatch.setattr(service._transport, "request", lost)
        first = training.step(records())
        assert entered.wait(5)
        second = training.step(records())
        release.set()
        with pytest.raises(TransportError):
            first.result(5)
        with pytest.raises(ConnectionError, match="not sent"):
            second.accepted().result(5)
        with pytest.raises(ConnectionError, match="reattach"):
            training.step(records())
        assert first.request_id != second.request_id and not run.workers.calls


def test_capacity_wait_retains_dependencies_and_impossible_snapshot_does_not_submit(
    tmp_path, monkeypatch
):
    with runtime(tmp_path) as (service, training, run, _):
        staging = training._snapshots
        training._snapshots = BufferBudget(1)
        with pytest.raises(ValueError, match="staging bytes"):
            training.step(records())
        assert training._next == 1 and run.journal.last_sequence == 0
        training._snapshots = staging
        training._slots = BufferBudget(1, 1)
        entered, release = Event(), Event()
        original = service._transport.upload

        def delay(*args):
            entered.set()
            assert release.wait(5)
            return original(*args)

        monkeypatch.setattr(service._transport, "upload", delay)
        failed = training.forward_backward(records(), loss_params={"temperature": -1})
        assert entered.wait(5)
        queued = []
        caller = Thread(target=lambda: queued.append(training.optim_step()))
        caller.start()
        until(training._submission.locked)
        release.set()
        caller.join(5)
        assert not caller.is_alive()
        for future in (failed, queued[0]):
            with pytest.raises(TrainingRequestError):
                future.result(5)
        assert run.journal.get(queued[0].request_id)[0].dependencies == (
            failed.request_id,
        )
        assert not run.workers.calls
