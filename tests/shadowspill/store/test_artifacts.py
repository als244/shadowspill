from __future__ import annotations

import errno
import json
import os
from pathlib import Path

import pytest

from shadowspill.pytorch.profiling.metadata import (
    canonicalize_profiling_metadata,
    training_profiling_metadata,
)
from shadowspill.schema import ARTIFACT_VERSION
from shadowspill.store import ArtifactStore
from shadowspill.store import artifacts as store_module


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
    assert cache.build == root / "build"
    assert cache.planning == root / "planning"
    assert cache.exports == root / "build" / "exports"
    assert cache.graphpairs == root / "build" / "graphpairs"
    assert cache.profile_measurements == root / "build" / "profiling" / "measurements"
    assert cache.compiled_manifests == (
        root / "build" / "profiling" / "compiled_manifests"
    )
    assert cache.programs_archive == root / "planning" / "programs"
    assert cache.plan_requests == root / "planning" / "requests"
    assert cache.plan_selections == root / "planning" / "results"
    assert cache.plans == root / "planning" / "plans"
    assert "mlops-build-17" in cache.inductor.name
    assert cache.inductor.parent == root / "build" / "inductor"
    assert (root / "layout.json").is_file()
    assert (root / "README.md").is_file()
    assert dict(cache.diagnostics())["build"] == str(root / "build")

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


def test_a_plan_store_roots_the_planning_kinds_apart(tmp_path: Path) -> None:
    cache = ArtifactStore.resolve(
        tmp_path / "shared", plan_store=tmp_path / "run" / "plan_store"
    )
    cache.initialize()

    shared = tmp_path / "shared" / f"v{ARTIFACT_VERSION}"
    plans = tmp_path / "run" / "plan_store" / f"v{ARTIFACT_VERSION}"
    assert cache.root == shared
    assert cache.plan_store == plans
    assert cache.build == shared / "build"
    assert cache.planning == plans / "planning"
    # what a run pays for stays shared
    assert cache.exports == shared / "build" / "exports"
    # the program a plan was for follows the plans, not the builds
    assert cache.programs_archive == plans / "planning" / "programs"
    # what a run measured is its own
    assert cache.plan_requests == plans / "planning" / "requests"
    assert cache.plan_selections == plans / "planning" / "results"
    assert cache.plans == plans / "planning" / "plans"
    assert (plans / "layout.json").is_file()
    assert (plans / "README.md").is_file()
    assert (shared / "layout.json").is_file()
    assert dict(cache.diagnostics())["plan_store"] == str(plans)

    single = ArtifactStore.resolve(tmp_path / "alone")
    assert single.plan_store is None
    assert single.planning == single.root / "planning"
    assert dict(single.diagnostics())["plan_store"] == str(single.root)


def test_the_home_cache_is_the_default_store() -> None:
    cache = ArtifactStore.resolve(None)

    assert cache.root == (Path.home() / ".cache" / "shadowspill").resolve() / (
        f"v{ARTIFACT_VERSION}"
    )
    assert cache.plan_store is None


def test_planning_cache_policy_flags_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        ArtifactStore.resolve(tmp_path, implementation_revision=" ")

    with pytest.raises(ValueError, match="must be one of"):
        ArtifactStore.resolve(tmp_path, build_store_mode="readonly")  # type: ignore[arg-type]

    # A run that contributes to neither tree leaves nothing behind at all.
    transient_root = tmp_path / "transient"
    transient = ArtifactStore.resolve(
        transient_root, build_store_mode="reuse", plan_store_mode="reuse"
    )
    with transient.activate_pytorch():
        assert not transient_root.exists()


def test_the_two_trees_are_rooted_and_permitted_apart(tmp_path: Path) -> None:
    """A build store several runs share, and a plan store each keeps.

    The two were one switch until they could be rooted apart, and one switch
    meant a run keeping its plans to itself also stopped contributing the
    builds it had paid for.
    """

    shared, mine = tmp_path / "shared", tmp_path / "mine"
    store = ArtifactStore.resolve(
        None,
        build_store=shared,
        plan_store=mine,
        build_store_mode="require",
        plan_store_mode="contribute",
    )
    # each tree under its own root, versioned as a store always is
    assert store.build_store is not None and shared in store.build_store.parents
    assert store.plan_store is not None and mine in store.plan_store.parents
    assert store.build == store.build_store / "build"
    assert store.planning == store.plan_store / "planning"
    assert not store.build_policy.write_enabled
    assert store.build_policy.require_hit
    assert store.plan_policy.write_enabled
    assert not store.plan_policy.require_hit

    # Keeping plans to itself no longer silences the build tree.
    keeps_plans = ArtifactStore.resolve(tmp_path / "both", plan_store_mode="reuse")
    assert keeps_plans.build_policy.write_enabled
    assert not keeps_plans.plan_policy.write_enabled


def test_a_refreshing_build_store_publishes_an_isolated_pytorch_cache(
    tmp_path: Path,
) -> None:
    """A refresh ignores what is there, works apart, and publishes at the end.

    The isolation is what makes `refresh` safe to run beside a store others
    are reading: nothing lands in it until the run has finished and is ready
    to replace what was there.
    """

    cache = ArtifactStore.resolve(
        tmp_path,
        build_store_mode="refresh",
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
