"""Fresh weights require owned complete provenance, never optimizer-resume state."""

import io
import subprocess
import sys
from dataclasses import replace

import pytest

from engine_tests.server.test_artifacts import reference, store
from engine_tests.server.test_checkpoint_restore import configuration
from shadowspill.server.startup.weights import resolve_weights
from shadowspill_client import Optimizer, Weights
from shadowspill_client.protocol.checkpoints import (
    WEIGHTS_MEDIA_TYPE,
    WEIGHTS_RANK_MEDIA_TYPE,
    SavedRank,
    SaveInfo,
    WeightsManifest,
)
from shadowspill_client.protocol.codec import encode
from shadowspill_client.protocol.frozen import FrozenBase


def generation(artifacts, *, base=None, byteorder=sys.byteorder):
    def put(name, body, media_type="application/octet-stream"):
        ref = replace(reference(body, name), media_type=media_type)
        artifacts.put(ref, lambda _: io.BytesIO(body))
        return ref

    config = configuration()
    info = SaveInfo(
        "session",
        "source-run",
        "history",
        "export",
        17,
        2,
        put("configuration", encode(config)),
        put("plans", b"source Plans"),
        byteorder,
        kind="weights",
        frozen_base=base,
        frozen_state_id=None if base is None else base.identity,
    )
    manifest = WeightsManifest(
        info,
        [
            SavedRank(
                rank,
                put(
                    f"rank-{rank}",
                    b"archive verified by reader",
                    WEIGHTS_RANK_MEDIA_TYPE,
                ),
                str(rank) * 64,
            )
            for rank in range(2)
        ],
    )
    return put("generation", encode(manifest), WEIGHTS_MEDIA_TYPE), manifest, config


def test_weights_resolution_allows_fresh_optimizer_and_keeps_source_provenance(
    tmp_path,
):
    with store(tmp_path) as artifacts:
        ref, manifest, config = generation(artifacts)
        target = replace(
            config,
            model=replace(config.model, initialize=Weights.artifact(ref.id)),
            optimizer=Optimizer("different/optimizer@1", defaults={"lr": 0.2}),
        )
        resolved = resolve_weights(artifacts, ref, target)
        assert resolved.reference == ref and resolved.manifest == manifest
        assert resolved.manifest.info.completed_updates == 17
        assert resolved.manifest.info.world_size == 2
        assert target.parallelism["num_replicas"] == 1
        assert target.optimizer != config.optimizer


@pytest.mark.parametrize("missing", ["generation", "configuration", "plans", "rank-1"])
def test_weights_resolution_requires_every_dependency_in_the_owner_namespace(
    tmp_path, monkeypatch, missing
):
    with store(tmp_path) as artifacts:
        ref, _, config = generation(artifacts)
        original = artifacts.get
        monkeypatch.setattr(
            artifacts, "get", lambda key: None if key == missing else original(key)
        )
        with pytest.raises(ValueError, match=r"owner|dependency"):
            resolve_weights(artifacts, ref, config)


def test_weights_resolution_rejects_other_kinds_and_byte_order(tmp_path):
    with store(tmp_path) as artifacts:
        ref, _, config = generation(
            artifacts, byteorder="big" if sys.byteorder == "little" else "little"
        )
        with pytest.raises(ValueError, match="byte order"):
            resolve_weights(artifacts, ref, config)
        with pytest.raises(ValueError, match="weights generation"):
            resolve_weights(artifacts, artifacts.get("configuration"), config)


def test_partial_weights_require_the_exact_frozen_model_and_artifact(
    tmp_path, monkeypatch
):
    with store(tmp_path) as artifacts:
        config = configuration()
        artifact = reference(b"base", "base")
        artifacts.put(artifact, lambda _: io.BytesIO(b"base"))
        base = FrozenBase(
            replace(config.model, initialize=Weights.artifact(artifact.id)),
            config.component_sources["model"],
            artifact,
        )
        ref, manifest, _ = generation(artifacts, base=base)
        assert resolve_weights(artifacts, ref, config).manifest == manifest
        with pytest.raises(ValueError, match="frozen base differs"):
            resolve_weights(
                artifacts,
                ref,
                replace(
                    config, model=replace(config.model, config={"different": True})
                ),
            )
        original = artifacts.get
        monkeypatch.setattr(
            artifacts, "get", lambda key: None if key == "base" else original(key)
        )
        with pytest.raises(ValueError, match="dependency"):
            resolve_weights(artifacts, ref, config)


def test_weights_generation_resolution_import_needs_no_framework_or_catalog():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
for name in ('torch', 'triton', 'mlops', 'shadowspill_workloads'):
    sys.modules[name] = None
from shadowspill.server.startup.weights import resolve_weights
""",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
