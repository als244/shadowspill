"""Checkpoint creation through public HTTP and the existing worker owner."""

import io
import sys
from dataclasses import replace

import pytest

from engine_tests.server.test_artifacts import reference
from engine_tests.server.test_checkpoints import stream_rank, workers
from engine_tests.server.test_http import TOKEN
from engine_tests.server.test_http_creation import endpoint
from engine_tests.server.test_jobs import until
from engine_tests.server.test_local_creation import configured
from engine_tests.server.test_service_client import address
from shadowspill.server.checkpoints import publish_checkpoint
from shadowspill.server.records.requests import RunIdentity
from shadowspill.server.startup import creation as local_creation
from shadowspill.server.startup.checkpoint import (
    CheckpointResolver,
    validate_resume_optimizer,
)
from shadowspill.server.startup.records import CreationRecord
from shadowspill.training.worker import runtime
from shadowspill_client import ServiceClient, Weights
from shadowspill_client.creation import CreationError
from shadowspill_client.protocol.checkpoints import CheckpointConversion, SaveInfo
from shadowspill_client.protocol.creation import TrainingCreation
from shadowspill_client.protocol.plans import summary_id


def checkpoint(ctx):
    creation = CreationRecord.encode(TrainingCreation(ctx.config))
    with ctx.service.artifacts("owner") as artifacts:
        creation.publish(artifacts)
        plans = ctx.saved
        if plans is None:
            plans = reference(b"source-plans", "source-plans")
            artifacts.put(plans, lambda _: io.BytesIO(b"source-plans"))
        info = SaveInfo(
            "previous-session",
            "source-run",
            "source-history",
            "save",
            7,
            1,
            creation.configuration,
            plans,
            sys.byteorder,
            metadata={"cursor": 19},
        )
        executors, ranks = workers(info)
        return publish_checkpoint(
            artifacts,
            info,
            ranks,
            lambda rank, sink: stream_rank(executors[rank.rank], info, sink),
        )


@pytest.mark.parametrize("reupload", [False, True])
def test_native_resume_reads_checkpoint_defaults_without_a_generated_plan_summary(
    tmp_path, monkeypatch, reupload
):
    def no_conversion(*args, **kwargs):
        pytest.fail("identical saved Plans must not convert checkpoint tensors")

    monkeypatch.setattr(local_creation.LocalCreation, "_convert_state", no_conversion)
    with configured(tmp_path, monkeypatch, saved=True) as ctx:
        source = checkpoint(ctx)
        with ctx.service.artifacts("owner") as artifacts:
            assert artifacts.get(summary_id(ctx.saved, 0)) is None
        with (
            endpoint(ctx) as (transport, _),
            ServiceClient(address(transport), token=TOKEN) as service,
        ):
            plans = service.artifact(ctx.saved)
            if reupload:
                path = plans.download(tmp_path / "saved-plan.ssp").result(5)
                plans = service.artifacts.upload(path).result(5)
                assert plans.reference.id != ctx.saved.id
                with ctx.service.artifacts("owner") as artifacts:
                    assert artifacts.get(summary_id(plans.reference, 0)) is None
            training = service.create_training_client(
                plans=plans, checkpoint=service.artifact(source)
            ).result(10)
            assert training.get_config() == ctx.config
            assert training.plans.reference == plans.reference
            assert ctx.owners[-1]._node.prepared[0].restore.checkpoint == source
            assert ctx.owners[-1].run.controller.boundary.completed_updates == 7
            assert not ctx.phases
            training.close().result(5)
            until(lambda: ctx.jobs.used == (0, 0))


@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("initial", [None, Weights.artifact("absent-fresh-weights")])
def test_resume_skips_fresh_initializer_and_preserves_ready_lineage(
    tmp_path,
    monkeypatch,
    saved,
    initial,
):
    def forbidden(*args, **kwargs):
        pytest.fail("checkpoint creation asked for fresh-weight initialization")

    # The fixture's child exercises startup/lifecycle, not model execution.
    # Its placeholder Plans cannot be converted. Actual conversion and archive
    # loading use real bundles in test_checkpoint_conversion and the GPU demo.
    conversions = []

    def convert(owner, source, plans, config, world_size, root, *, wait):
        assert source.manifest.info.plans != plans
        assert world_size == source.manifest.info.world_size
        info = replace(
            source.manifest.info,
            plans=plans,
            configuration=owner.info.configuration,
            request_id="converted-" + owner.identity.run_id,
            conversion=CheckpointConversion(
                source.reference, source.manifest.info.plans, "a" * 64
            ),
        )
        executors, ranks = workers(info)
        reference = publish_checkpoint(
            owner._artifacts,
            info,
            ranks,
            lambda rank, sink: stream_rank(executors[rank.rank], info, sink),
        )
        resolved = CheckpointResolver(
            owner._artifacts,
            session_id=info.session_id,
            world_size=world_size,
            configuration=config,
        ).resolve(reference)
        conversions.append(reference)
        return resolved

    monkeypatch.setattr(local_creation.LocalCreation, "_convert_state", convert)
    with configured(
        tmp_path,
        monkeypatch,
        saved=saved,
        change=lambda c: replace(c, model=replace(c.model, initialize=initial)),
    ) as ctx:
        source = checkpoint(ctx)
        monkeypatch.setattr(local_creation, "prepare_initialization", forbidden)
        monkeypatch.setattr(runtime, "prepare_initialization", forbidden)
        with (
            endpoint(ctx) as (transport, _),
            ServiceClient(
                address(transport), token=TOKEN, session_id="resume-session"
            ) as service,
        ):
            training = service.create_training_client(
                config=ctx.config, plans=ctx.saved, checkpoint=source
            ).result(10)
            identity = RunIdentity("owner", service.session_id, training.run_id)
            info = ctx.service.catalog.get(identity)
            assert info.parent_checkpoint == source
            initial_checkpoint = source if saved else conversions[-1]
            assert info.initial_checkpoint == initial_checkpoint
            assert info.initial_completed_updates == 7
            assert info.initial_history_id != "source-history"
            assert training.get_config() == ctx.config
            owner = ctx.owners[-1]
            worker = owner._node.prepared[0]
            assert worker.initialization is None
            assert worker.restore.checkpoint == initial_checkpoint
            assert owner._node._launches[0].weights == ()
            assert owner.run.controller.boundary.completed_updates == 7
            closed = training.close().result(5)
            assert (
                closed.completed_updates == 7
                and closed.history_id == info.initial_history_id
            )
            until(lambda: ctx.jobs.used == (0, 0))
            assert ctx.service.catalog.get(identity).parent_checkpoint == source
            phases = list(ctx.phases)

            # Only immutable artifacts are supplied; the SDK reads bounded
            # configuration metadata, without downloading or decoding Plans.
            again = service.create_training_client(
                plans=training.plans, checkpoint=source
            ).result(10)
            assert again.get_config() == ctx.config
            assert again.run_id != training.run_id and ctx.phases == phases
            second = again.close().result(5)
            assert (
                second.completed_updates == 7 and second.history_id != closed.history_id
            )
            until(lambda: ctx.jobs.used == (0, 0))

            assert len(conversions) == (0 if saved else 2)


def test_failed_checkpoint_delivery_never_publishes_initialized_lineage(
    tmp_path, monkeypatch
):
    with configured(tmp_path, monkeypatch, saved=True) as ctx:
        source = checkpoint(ctx)
        with ctx.service.artifacts("owner") as artifacts:
            from shadowspill_client.protocol.codec import decode

            with artifacts.open(source) as stream:
                manifest = decode(stream.read())
            artifacts._path(manifest.ranks[0].archive).write_bytes(b"corrupt")
        with (
            endpoint(ctx) as (transport, _),
            ServiceClient(
                address(transport), token=TOKEN, session_id="resume-session"
            ) as service,
        ):
            pending = service.create_training_client(
                config=ctx.config, plans=ctx.saved, checkpoint=source
            )
            with pytest.raises(CreationError):
                pending.result(10)
            until(lambda: ctx.jobs.used == (0, 0))
            info = ctx.service.catalog.get(
                RunIdentity("owner", service.session_id, pending.run_id)
            )
            assert info.state == "failed" and info.parent_checkpoint is None
            assert (
                info.initial_history_id is None and info.initial_completed_updates == 0
            )
            assert ctx.owners[-1].closed and not ctx.phases
            assert "payload" in info.error.message


@pytest.mark.parametrize("change", ["defaults", "objective", "semantics", "source"])
def test_new_setup_preserves_optimizer_interpretation_but_not_old_defaults(
    tmp_path, monkeypatch, change
):
    with configured(tmp_path, monkeypatch, saved=True) as ctx:
        reference = checkpoint(ctx)
        config = ctx.config
        if change == "defaults":
            config = replace(
                config, optimizer=replace(config.optimizer, defaults={"lr": 0.2})
            )
        elif change == "objective":
            config = replace(
                config,
                objective=replace(config.objective, definition="different-objective"),
            )
        elif change == "semantics":
            config = replace(
                config,
                optimizer=replace(config.optimizer, config={"reinterpret_state": True}),
            )
        else:
            config = replace(
                config,
                definition_sources={
                    **config.definition_sources,
                    "optimizer": replace(
                        config.definition_sources["optimizer"], version="changed"
                    ),
                },
            )
        with ctx.service.artifacts("owner") as artifacts:
            source = CheckpointResolver(
                artifacts, session_id="unused", world_size=1, configuration=ctx.config
            ).resolve(reference)
            if change in ("semantics", "source"):
                with pytest.raises(ValueError, match="optimizer semantics"):
                    validate_resume_optimizer(artifacts, source, config)
            else:
                validate_resume_optimizer(artifacts, source, config)
