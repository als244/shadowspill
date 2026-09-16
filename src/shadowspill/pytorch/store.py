"""The artifact store's framework half: exported programs and compiler caches.

`shadowspill.store` owns content addressing, atomic writes, digests and the four
modes that gate reading and writing. Two kinds of artifact need PyTorch to handle
them at all, and they are here:

- the exported program archive, which only `torch.export` can write and read;
- the compiler's own cache tree, which is managed by the compiler rather than by
  us, and whose lookups are routed through process-global environment variables.

A `FrameworkArtifacts` wraps a store and adds those two. A second frontend writes
its own, and the neutral store needs no callback, no optional parameter and no
knowledge that either exists.
"""

from __future__ import annotations

import errno
import hashlib
import os
import shutil
import tempfile
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

from shadowspill.schema import artifact_schema
from shadowspill.store import ArtifactStore
from shadowspill.store.artifacts import (
    atomic_json,
    digest_directory,
    read_json,
    safe_label,
)

_PYTORCH_CACHE_ENVIRONMENT = "TORCHINDUCTOR_CACHE_DIR"
_CACHE_ENVIRONMENT_LOCK = threading.RLock()
_LAYOUT_SCHEMA = artifact_schema("artifact_store")
_EXPORT_SCHEMA = artifact_schema("pytorch.export")


class FrameworkArtifacts:
    """One store, plus the two artifact kinds only a framework can handle."""

    __slots__ = ("store",)

    def __init__(self, store: ArtifactStore) -> None:
        self.store = store

    @property
    def compiler_cache(self) -> Path:
        revision = self.store.export_bypass_key or "default"
        identity = hashlib.sha256(revision.encode()).hexdigest()[:12]
        return self.store.build / "inductor" / f"{safe_label(revision)}-{identity}"

    @contextmanager
    def activate(self) -> Iterator[None]:
        """Route process-global compiler cache lookups for one planning call."""

        if self.store.build_writes_enabled or self.store.plan_policy.write_enabled:
            self.store.initialize()
        with _CACHE_ENVIRONMENT_LOCK:
            previous = os.environ.get(_PYTORCH_CACHE_ENVIRONMENT)
            previous_triton = os.environ.get("TRITON_CACHE_DIR")
            isolated = (
                not self.store.build_reads_enabled
                or not self.store.build_writes_enabled
            )
            with tempfile.TemporaryDirectory(
                prefix="shadowspill-plan-",
            ) as temporary:
                active = Path(temporary) if isolated else self.compiler_cache
                clear_caches = _clear_compiler_caches if isolated else None
                if clear_caches is not None:
                    clear_caches()
                os.environ[_PYTORCH_CACHE_ENVIRONMENT] = str(active)
                os.environ["TRITON_CACHE_DIR"] = str(active / "triton")
                completed = False
                try:
                    yield
                    completed = True
                finally:
                    if clear_caches is not None:
                        clear_caches()
                    if previous is None:
                        os.environ.pop(_PYTORCH_CACHE_ENVIRONMENT, None)
                    else:
                        os.environ[_PYTORCH_CACHE_ENVIRONMENT] = previous
                    if previous_triton is None:
                        os.environ.pop("TRITON_CACHE_DIR", None)
                    else:
                        os.environ["TRITON_CACHE_DIR"] = previous_triton

                if completed and self.store.build_writes_enabled and isolated:
                    _publish_cache_tree(
                        active,
                        self.compiler_cache,
                        overwrite=self.store.build_policy.overwrite,
                    )
                if self.store.build_writes_enabled:
                    self.store.record(
                        category="pytorch",
                        kind="inductor_cache",
                        digest=None,
                        path=self.compiler_cache,
                        access="managed",
                        schema=None,
                    )

    def archive_export(
        self,
        exported_program: Any,
        *,
        digest: str,
        metadata: Mapping[str, object],
    ) -> Path:
        """Persist a freshly produced Export artifact and readable manifest.

        An existing identical archive is *matched*, not loaded.  Skipping the
        Export call requires a separately trusted pre-capture identity; this
        archive alone never guesses Python objective semantics.
        """

        directory = digest_directory(self.store.exports, digest)
        artifact_path = directory / "exported_program.pt2"
        manifest_path = directory / "manifest.json"
        if not self.store.build_writes_enabled:
            return artifact_path
        if self._match_export_archive(
            directory,
            artifact_path,
            manifest_path,
            digest,
        ):
            return artifact_path
        self._write_export_archive(
            exported_program,
            artifact_path,
            manifest_path,
            digest,
            metadata,
        )
        self._record_export_archive(artifact_path, manifest_path, digest, "write")
        return artifact_path

    def _match_export_archive(
        self,
        directory: Path,
        artifact_path: Path,
        manifest_path: Path,
        digest: str,
    ) -> bool:
        if (
            not artifact_path.exists()
            or not manifest_path.exists()
            or self.store.build_policy.overwrite
        ):
            return False
        manifest = read_json(manifest_path)
        if manifest.get("schema") != _EXPORT_SCHEMA or manifest.get("digest") != digest:
            raise ValueError(f"Export cache entry {directory} is invalid")
        self._record_export_archive(artifact_path, manifest_path, digest, "matched")
        return True

    @staticmethod
    def _write_export_archive(
        exported_program: Any,
        artifact_path: Path,
        manifest_path: Path,
        digest: str,
        metadata: Mapping[str, object],
    ) -> None:
        import torch

        directory = artifact_path.parent
        directory.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".exported_program.", suffix=".pt2", dir=directory
        )
        os.close(descriptor)
        try:
            torch.export.save(exported_program, temporary)
            os.replace(temporary, artifact_path)
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temporary)
        atomic_json(
            manifest_path,
            {
                "schema": _EXPORT_SCHEMA,
                "digest": digest,
                "artifact": artifact_path.name,
                "metadata": dict(metadata),
            },
        )

    def _record_export_archive(
        self,
        artifact_path: Path,
        manifest_path: Path,
        digest: str,
        artifact_access: str,
    ) -> None:
        if artifact_access == "matched":
            self.store.record(
                category="pytorch",
                kind="export_manifest",
                digest=digest,
                path=manifest_path,
                access="read",
                schema=_EXPORT_SCHEMA,
            )
        self.store.record(
            category="pytorch",
            kind="exported_program",
            digest=digest,
            path=artifact_path,
            access=artifact_access,
            schema=_EXPORT_SCHEMA,
        )
        if artifact_access == "matched":
            return
        self.store.record(
            category="pytorch",
            kind="export_manifest",
            digest=digest,
            path=manifest_path,
            access="write",
            schema=_EXPORT_SCHEMA,
        )


def _clear_compiler_caches() -> None:
    """Clear process-local compiler state at an isolated cache boundary."""

    # The framework is version-pinned.  This private helper is
    # deliberately confined here; it prevents an earlier plan in this process
    # from reading entries a refresh was told to ignore.
    from torch._inductor.utils import clear_caches

    clear_caches()


def _publish_cache_tree(source: Path, destination: Path, *, overwrite: bool) -> None:
    """Publish a fresh, write-enabled compiler cache without replaying old data."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        try:
            os.replace(source, destination)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            _copy_cache_tree_atomically(source, destination)
        return

    for source_path in sorted(source.rglob("*")):
        relative = source_path.relative_to(source)
        destination_path = destination / relative
        if source_path.is_dir():
            destination_path.mkdir(parents=True, exist_ok=True)
            continue
        if source_path.name.endswith(".lock"):
            continue
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if destination_path.exists():
            if _files_equal(source_path, destination_path):
                continue
            if not overwrite:
                raise ValueError(
                    "fresh compiler artifact conflicts with an existing "
                    "entry; use a 'refresh' store mode or a new "
                    f"export_bypass_key: {destination_path}"
                )
        temporary = destination_path.with_name(
            f".{destination_path.name}.{os.getpid()}.tmp"
        )
        with suppress(FileNotFoundError):
            temporary.unlink()
        try:
            try:
                os.link(source_path, temporary)
            except OSError:
                shutil.copy2(source_path, temporary)
            os.replace(temporary, destination_path)
        finally:
            with suppress(FileNotFoundError):
                temporary.unlink()


def _copy_cache_tree_atomically(source: Path, destination: Path) -> None:
    """Publish a cache tree across filesystems through a sibling staging path."""

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.{os.getpid()}.",
            dir=destination.parent,
        )
    )
    try:
        shutil.copytree(source, staging, dirs_exist_ok=True)
        os.replace(staging, destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _files_equal(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as left_file, right.open("rb") as right_file:
        while True:
            left_chunk = left_file.read(1 << 20)
            right_chunk = right_file.read(1 << 20)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True
