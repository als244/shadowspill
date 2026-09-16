"""The artifact store's framework half: the compiler cache and the archive."""

from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest

from shadowspill.pytorch import store as store_module
from shadowspill.pytorch.store import FrameworkArtifacts
from shadowspill.store import ArtifactStore


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
        export_bypass_key="fresh-cache-test",
    )
    previous = os.environ.get("TORCHINDUCTOR_CACHE_DIR")

    with FrameworkArtifacts(cache).activate():
        active = Path(os.environ["TORCHINDUCTOR_CACHE_DIR"])
        assert active != FrameworkArtifacts(cache).compiler_cache
        assert not FrameworkArtifacts(cache).compiler_cache.exists()
        marker = active / "fxgraph" / "test" / "artifact"
        marker.parent.mkdir(parents=True)
        marker.write_text("indexed")

    compiler_cache = FrameworkArtifacts(cache).compiler_cache
    assert (compiler_cache / "fxgraph" / "test" / "artifact").read_text() == "indexed"
    assert os.environ.get("TORCHINDUCTOR_CACHE_DIR") == previous
    assert any(
        artifact.kind == "inductor_cache" and artifact.path == compiler_cache
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


def test_planning_cache_policy_flags_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        ArtifactStore.resolve(tmp_path, export_bypass_key=" ")

    with pytest.raises(ValueError, match="must be one of"):
        ArtifactStore.resolve(tmp_path, build_store_mode="readonly")  # type: ignore[arg-type]

    # A run that contributes to neither tree leaves nothing behind at all.
    transient_root = tmp_path / "transient"
    transient = ArtifactStore.resolve(
        transient_root, build_store_mode="reuse", plan_store_mode="reuse"
    )
    with FrameworkArtifacts(transient).activate():
        assert not transient_root.exists()
