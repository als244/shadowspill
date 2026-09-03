from __future__ import annotations

import errno
import json
import os
from pathlib import Path

import pytest

from shadowspill.planner import artifact_store as store_module
from shadowspill.planner.artifact_store import ArtifactStore
from shadowspill.pytorch.profiling.metadata import (
    canonicalize_profiling_metadata,
    training_profiling_metadata,
)
from shadowspill.schema import ARTIFACT_VERSION


def test_profiling_metadata_is_canonical_and_position_aligned() -> None:
    first = canonicalize_profiling_metadata(
        {"sequence_lengths": [512, 512], "sequence_count": 2}
    )
    reordered = canonicalize_profiling_metadata(
        {"sequence_count": 2, "sequence_lengths": [512, 512]}
    )
    different = canonicalize_profiling_metadata(
        {"sequence_lengths": [1024], "sequence_count": 1}
    )

    assert first == reordered
    assert first.digest != different.digest
    assert json.loads(first.canonical_json)["value"]["sequence_count"] == 2
    assert training_profiling_metadata([None, {"tokens": 8}], microbatch_count=2)

    with pytest.raises(ValueError, match="one entry per example microbatch"):
        training_profiling_metadata([None], microbatch_count=2)
    with pytest.raises(ValueError, match="finite"):
        canonicalize_profiling_metadata({"value": float("nan")})
    with pytest.raises(TypeError, match="JSON-compatible"):
        canonicalize_profiling_metadata({"value": object()})


def test_planning_cache_has_stable_human_readable_layout(tmp_path: Path) -> None:
    cache = ArtifactStore.resolve(
        tmp_path,
        implementation_revision="mlops-build-17",
    )
    cache.initialize()

    root = tmp_path / f"v{ARTIFACT_VERSION}"
    assert cache.root == root
    assert cache.pytorch == root / "pytorch"
    assert cache.graphpairs == root / "graphpairs"
    assert cache.profiling == root / "profiling"
    assert cache.pressurefit == root / "pressurefit"
    assert cache.plans == root / "plans"
    assert cache.profile_measurements == root / "profiling" / "measurements"
    assert cache.pressurefit_selections == root / "pressurefit" / "selections"
    assert "mlops-build-17" in cache.inductor.name
    assert (root / "layout.json").is_file()
    assert (root / "README.md").is_file()

    digest = "a" * 64
    cache.record(
        category="profiling",
        kind="task_measurement",
        digest=digest,
        path=tmp_path / "measurement.json",
        access="write",
        schema="test/v1",
    )
    cache.record(
        category="profiling",
        kind="task_measurement",
        digest=digest,
        path=tmp_path / "measurement.json",
        access="write",
        schema="test/v1",
    )
    assert len(cache.artifacts()) == 1


def test_planning_cache_policy_flags_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires"):
        ArtifactStore.resolve(tmp_path, overwrite_plan=True)
    with pytest.raises(ValueError, match="non-empty"):
        ArtifactStore.resolve(tmp_path, implementation_revision=" ")

    transient_root = tmp_path / "transient"
    transient = ArtifactStore.resolve(transient_root, save_plan=False)
    with transient.activate_pytorch():
        assert not transient_root.exists()


def test_force_fresh_publishes_a_write_enabled_isolated_pytorch_cache(
    tmp_path: Path,
) -> None:
    cache = ArtifactStore.resolve(
        tmp_path,
        force_fresh=True,
        implementation_revision="fresh-cache-test",
    )
    previous = os.environ.get("TORCHINDUCTOR_CACHE_DIR")

    with cache.activate_pytorch():
        active = Path(os.environ["TORCHINDUCTOR_CACHE_DIR"])
        assert active != cache.inductor
        assert not cache.inductor.exists()
        marker = active / "fxgraph" / "test" / "artifact"
        marker.parent.mkdir(parents=True)
        marker.write_text("indexed")

    assert (cache.inductor / "fxgraph" / "test" / "artifact").read_text() == ("indexed")
    assert os.environ.get("TORCHINDUCTOR_CACHE_DIR") == previous
    assert any(
        artifact.kind == "inductor_cache" and artifact.path == cache.inductor
        for artifact in cache.artifacts()
    )


def test_inductor_cache_publish_crosses_filesystems_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "published"
    artifact = source / "triton" / "kernel"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("indexed")
    replace = os.replace
    calls = 0

    def cross_device_once(left: object, right: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError(errno.EXDEV, "cross-device link")
        replace(left, right)

    monkeypatch.setattr(store_module.os, "replace", cross_device_once)
    store_module._publish_cache_tree(source, destination, overwrite=False)

    assert (destination / "triton" / "kernel").read_text() == "indexed"
    assert source.is_dir()
