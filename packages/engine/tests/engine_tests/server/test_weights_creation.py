"""Service setup binds converted weights to fresh CPU sentinel workers."""

import hashlib
import io
import sys
from dataclasses import replace

import pytest

from engine_tests.server.test_jobs import until
from engine_tests.server.test_local_creation import configured, submit
from engine_tests.server.test_storage import IDENTITY
from shadowspill.orchestration.launch.file_grants import verify_file
from shadowspill.server import local_creation as module
from shadowspill.server.weights_initialization import ResolvedWeights
from shadowspill_client import Weights
from shadowspill_client.protocol.checkpoints import (
    WEIGHTS_MEDIA_TYPE,
    WEIGHTS_RANK_MEDIA_TYPE,
    SavedRank,
    SaveInfo,
    WeightsManifest,
)
from shadowspill_client.protocol.codec import encode
from shadowspill_client.protocol.requests import ArtifactRef


def selected(config):
    return replace(
        config, model=replace(config.model, initialize=Weights.artifact("exported"))
    )


def publish(artifacts, identity, body, media_type="application/octet-stream"):
    reference = ArtifactRef(
        identity, hashlib.sha256(body).hexdigest(), len(body), media_type
    )
    artifacts.put(reference, lambda _: io.BytesIO(body))
    return reference


def exported(ctx, *, missing_rank=False):
    with ctx.service.artifacts("owner") as artifacts:
        configuration = publish(artifacts, "source-config", encode(ctx.config))
        rank = ArtifactRef("rank", "a" * 64, 1, WEIGHTS_RANK_MEDIA_TYPE)
        if not missing_rank:
            rank = publish(
                artifacts, "rank", b"CPU sentinel rank archive", WEIGHTS_RANK_MEDIA_TYPE
            )
        info = SaveInfo(
            "old-session",
            "old-run",
            "old-history",
            "save",
            17,
            1,
            configuration,
            ctx.saved,
            sys.byteorder,
            kind="weights",
        )
        manifest = WeightsManifest(info, (SavedRank(0, rank, "b" * 64),))
        reference = publish(artifacts, "exported", encode(manifest), WEIGHTS_MEDIA_TYPE)
        return ResolvedWeights(reference, manifest)


def test_owned_export_bypasses_model_importer_and_starts_at_zero_updates(
    tmp_path, monkeypatch
):
    with configured(tmp_path, monkeypatch, change=selected, saved=True) as ctx:
        source = exported(ctx)
        converted = []

        def convert(owner, saved, plans, config, world_size, root, *, wait):
            assert not ctx.node.used.devices and not ctx.phases
            assert saved == source and plans == ctx.saved and config == ctx.config
            assert world_size == 1 and not wait
            info = replace(saved.manifest.info, configuration=owner.info.configuration)
            manifest = replace(saved.manifest, info=info)
            reference = publish(
                owner._artifacts, "converted", encode(manifest), WEIGHTS_MEDIA_TYPE
            )
            result = ResolvedWeights(reference, manifest)
            converted.append(result)
            return result

        monkeypatch.setattr(module.LocalCreation, "_convert_state", convert)
        monkeypatch.setattr(
            module,
            "prepare_initialization",
            lambda _: pytest.fail("installed importer called for complete export"),
        )
        submit(ctx)
        until(lambda: ctx.service.catalog.get(IDENTITY).state != "preparing")
        assert ctx.service.catalog.get(IDENTITY).state == "ready", (
            ctx.service.catalog.get(IDENTITY)
        )
        until(lambda: next(iter(ctx.jobs._jobs.values())).live)
        owner = ctx.owners[0]
        worker = owner._node.prepared[0]
        assert len(converted) == 1
        assert worker.restore is worker.initialization is worker.frozen_base is None
        assert worker.initial_weights.artifact == source.reference
        assert worker.initial_weights.generation == converted[0].reference
        assert worker.initial_weights.info.completed_updates == 17
        assert owner.run.controller.boundary.completed_updates == 0
        launch = owner._node._launches[0]
        assert launch.weights == () and launch.initial_weights == worker.initial_weights
        path = verify_file(
            owner._node._grants.root, worker.initial_weights.rank.archive
        )
        assert path.read_bytes() == b"CPU sentinel rank archive"


@pytest.mark.parametrize("fault", ["manifest", "missing_rank", "custom_format"])
def test_artifact_source_preflight_fails_before_phases_and_preserves_custom_importers(
    tmp_path, monkeypatch, fault
):
    with configured(tmp_path, monkeypatch, change=selected, saved=True) as ctx:
        if fault == "missing_rank":
            exported(ctx, missing_rank=True)
        else:
            with ctx.service.artifacts("owner") as artifacts:
                publish(
                    artifacts,
                    "exported",
                    b"custom payload",
                    WEIGHTS_MEDIA_TYPE
                    if fault == "manifest"
                    else "application/x-custom-weights",
                )
        submit(ctx)
        until(lambda: ctx.jobs.used == (0, 0))
        info = ctx.service.catalog.get(IDENTITY)
        assert info.state == "failed"
        assert not ctx.phases and ctx.owners[0]._directory is None
        if fault == "custom_format":
            assert (
                "classifier fixture requires explicit random weights"
                in info.error.message
            )
        elif fault == "missing_rank":
            assert "dependency is unavailable" in info.error.message
