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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refresh ignores what is there, works apart, and publishes at the end.

    The isolation is what makes `refresh` safe to run beside a store others
    are reading: nothing lands in it until the run has finished and is ready
    to replace what was there.
    """

    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "node-local"))
    cache = ArtifactStore.resolve(
        tmp_path / "persistent",
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
    assert not compiler_cache.is_relative_to(cache.root)
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


def test_compiler_cache_is_shared_outside_rank_stores_and_nested_activation(
    tmp_path, monkeypatch
):
    root = tmp_path / "node-local"
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(root))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "original-triton"))
    first = FrameworkArtifacts(ArtifactStore.resolve(tmp_path / "rank-0"))
    second = FrameworkArtifacts(ArtifactStore.resolve(tmp_path / "rank-1"))
    expected = first.compiler_cache
    assert expected == second.compiler_cache
    assert expected.is_relative_to(root)
    with first.activate():
        assert Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]) == expected
        nested = FrameworkArtifacts(ArtifactStore.resolve(tmp_path / "nested"))
        assert nested.compiler_cache == expected
        with nested.activate():
            assert Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]) == expected
        assert Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]) == expected
        marker = Path(os.environ["TRITON_CACHE_DIR"]) / "kernel.ptx"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("compiled")
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(root)
    assert os.environ["TRITON_CACHE_DIR"] == str(tmp_path / "original-triton")
    assert not (first.store.build / "inductor").exists()
    assert not (second.store.build / "inductor").exists()


def test_cache_environment_restored_after_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "node-local"))
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    artifacts = FrameworkArtifacts(ArtifactStore.resolve(tmp_path / "persistent"))
    with pytest.raises(RuntimeError, match="failed"), artifacts.activate():
        raise RuntimeError("failed")
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(tmp_path / "node-local")
    assert "TRITON_CACHE_DIR" not in os.environ
    assert artifacts.compiler_cache.is_relative_to(tmp_path / "node-local")


def test_default_compiler_cache_uses_temporary_storage_without_launch_overrides(
    tmp_path, monkeypatch
):
    from torch._inductor.runtime import cache_dir_utils

    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    default = tmp_path / "tmp" / "torchinductor_user"
    monkeypatch.setattr(cache_dir_utils, "default_cache_dir", lambda: str(default))
    artifacts = FrameworkArtifacts(ArtifactStore.resolve(tmp_path / "cold-storage"))
    with artifacts.activate():
        active = Path(os.environ["TORCHINDUCTOR_CACHE_DIR"])
        assert active.is_relative_to(default)
        assert not active.is_relative_to(artifacts.store.root)
        assert Path(os.environ["TRITON_CACHE_DIR"]) == active / "triton"
    assert "TORCHINDUCTOR_CACHE_DIR" not in os.environ
    assert "TRITON_CACHE_DIR" not in os.environ


@pytest.mark.parametrize("cross_device", [False, True])
def test_ranks_can_publish_isolated_caches_to_a_shared_destination(
    tmp_path, monkeypatch, cross_device
):
    source, destination = tmp_path / "source", tmp_path / "cache"
    source.mkdir()
    (source / "this-rank.ptx").write_text("this kernel")
    replace = os.replace
    calls = 0

    def competing_publication(left, right):
        nonlocal calls
        calls += 1
        if cross_device and calls == 1:
            raise OSError(errno.EXDEV, "cross-device link")
        if calls == (2 if cross_device else 1):
            destination.mkdir()
            (destination / "other-rank.ptx").write_text("peer kernel")
            raise OSError(errno.ENOTEMPTY, "peer already published")
        replace(left, right)

    monkeypatch.setattr(store_module.os, "replace", competing_publication)
    store_module._publish_cache_tree(source, destination, overwrite=True)
    assert (destination / "this-rank.ptx").read_text() == "this kernel"
    assert (destination / "other-rank.ptx").read_text() == "peer kernel"
