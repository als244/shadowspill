"""Central, explicit, and inspectable artifact store for PyTorch planning."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shadowspill.ir import ExecutionPlan, Program
from shadowspill.ir.program import PROGRAM_SCHEMA
from shadowspill.schema import ARTIFACT_VERSION, artifact_schema

_PYTORCH_CACHE_ENVIRONMENT = "TORCHINDUCTOR_CACHE_DIR"
_CACHE_ENVIRONMENT_LOCK = threading.RLock()
_LAYOUT_SCHEMA = artifact_schema("artifact_store")
_EXPORT_SCHEMA = artifact_schema("pytorch.export")
_PLAN_MANIFEST_SCHEMA = artifact_schema("plan_manifest")
_ACCESS_KINDS = {"managed", "matched", "read", "write"}


@dataclass(frozen=True, slots=True)
class PlanningArtifact:
    """One file or managed directory touched by a planning call."""

    category: str
    kind: str
    digest: str | None
    path: Path
    access: str
    schema: str | None = None
    dependencies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.category or not self.kind:
            raise ValueError("planning artifact names must be non-empty")
        if self.digest is not None and len(self.digest) != 64:
            raise ValueError("planning artifact digest must be SHA-256")
        if self.access not in _ACCESS_KINDS:
            raise ValueError(f"unsupported planning artifact access {self.access!r}")


class _ArtifactLedger:
    """Thread-safe, insertion-ordered evidence for one planning call."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: list[PlanningArtifact] = []
        self._seen: set[PlanningArtifact] = set()

    def append(self, record: PlanningArtifact) -> None:
        with self._lock:
            if record in self._seen:
                return
            self._seen.add(record)
            self._records.append(record)

    def snapshot(self) -> tuple[PlanningArtifact, ...]:
        with self._lock:
            return tuple(self._records)


@dataclass(frozen=True, slots=True)
class ArtifactStore:
    """Where one planning call reads and writes its artifacts.

    Two trees under one versioned root. ``build`` holds what a run pays for
    and another run can reuse: exports, Inductor caches, graph pairs,
    profiles and lowered programs. ``planning`` holds what a run measured:
    the requests put to PressureFit, its results, and the plans callables
    run, the last a readable index linking one planning call to the
    immutable artifacts behind it. Content-addressed leaf paths provide
    identity throughout.

    ``plan_store`` puts the ``planning`` tree under another directory, so
    several runs can share one artifact store and each keep its own plans;
    ``None`` keeps both trees under ``root``.
    """

    root: Path
    build: Path
    planning: Path
    save_plan: bool = True
    force_fresh: bool = False
    overwrite_plan: bool = False
    implementation_revision: str | None = None
    plan_store: Path | None = None
    _ledger: _ArtifactLedger = field(
        default_factory=_ArtifactLedger,
        repr=False,
        compare=False,
    )

    @classmethod
    def resolve(
        cls,
        value: Any | None,
        *,
        plan_store_dir: Any | None = None,
        save_plan: bool = True,
        force_fresh: bool = False,
        overwrite_plan: bool = False,
        implementation_revision: str | None = None,
    ) -> ArtifactStore:
        for name, selected in (
            ("save_plan", save_plan),
            ("force_fresh", force_fresh),
            ("overwrite_plan", overwrite_plan),
        ):
            if not isinstance(selected, bool):
                raise TypeError(f"{name} must be a bool")
        if overwrite_plan and (not save_plan or not force_fresh):
            raise ValueError(
                "overwrite_plan=True requires save_plan=True and force_fresh=True"
            )
        if implementation_revision is not None:
            if not isinstance(implementation_revision, str):
                raise TypeError("implementation_revision must be a string or None")
            implementation_revision = implementation_revision.strip()
            if not implementation_revision:
                raise ValueError("implementation_revision must be non-empty")
        root = (
            _store_root(value, "artifact_store_dir")
            if value is not None
            else (Path.home() / ".cache" / "shadowspill").resolve()
            / f"v{ARTIFACT_VERSION}"
        )
        plan_store = (
            None
            if plan_store_dir is None
            else _store_root(plan_store_dir, "plan_store_dir")
        )
        return cls(
            root,
            root / "build",
            (root if plan_store is None else plan_store) / "planning",
            save_plan,
            force_fresh,
            overwrite_plan,
            implementation_revision,
            plan_store=plan_store,
        )

    @property
    def exports(self) -> Path:
        return self.build / "exports"

    @property
    def inductor(self) -> Path:
        revision = self.implementation_revision or "default"
        identity = hashlib.sha256(revision.encode()).hexdigest()[:12]
        return self.build / "inductor" / f"{_safe_label(revision)}-{identity}"

    @property
    def graphpairs(self) -> Path:
        return self.build / "graphpairs"

    @property
    def profiling(self) -> Path:
        return self.build / "profiling"

    @property
    def profile_measurements(self) -> Path:
        return self.profiling / "measurements"

    @property
    def compiled_manifests(self) -> Path:
        return self.profiling / "compiled_manifests"

    @property
    def pressurefit_programs(self) -> Path:
        return self.build / "programs"

    @property
    def pressurefit_requests(self) -> Path:
        return self.planning / "requests"

    @property
    def pressurefit_selections(self) -> Path:
        return self.planning / "results"

    @property
    def plans(self) -> Path:
        return self.planning / "plans"

    @property
    def read_enabled(self) -> bool:
        return not self.force_fresh

    @property
    def write_enabled(self) -> bool:
        return self.save_plan

    @contextmanager
    def activate_pytorch(self) -> Iterator[None]:
        """Route process-global Inductor cache lookups for one planning call."""

        if self.save_plan:
            self.initialize()
        with _CACHE_ENVIRONMENT_LOCK:
            previous = os.environ.get(_PYTORCH_CACHE_ENVIRONMENT)
            previous_triton = os.environ.get("TRITON_CACHE_DIR")
            isolated = self.force_fresh or not self.save_plan
            with tempfile.TemporaryDirectory(
                prefix="shadowspill-plan-",
            ) as temporary:
                active = Path(temporary) if isolated else self.inductor
                clear_caches = _clear_pytorch_compiler_caches if isolated else None
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

                if completed and self.save_plan and isolated:
                    _publish_cache_tree(
                        active,
                        self.inductor,
                        overwrite=self.overwrite_plan,
                    )
                if self.save_plan:
                    self.record(
                        category="pytorch",
                        kind="inductor_cache",
                        digest=None,
                        path=self.inductor,
                        access="managed",
                        schema=None,
                    )

    def initialize(self) -> None:
        """Create the stable top-level layout and its human guide.

        A store laid out before the ``build``/``planning`` split is moved
        into place first, directory by directory, so nothing in it is paid
        for again.
        """

        self.root.mkdir(parents=True, exist_ok=True)
        moved = _migrate_layout(self.root)
        for directory in (self.build, self.profiling, self.planning):
            directory.mkdir(parents=True, exist_ok=True)
        _write_guides(
            self.root,
            moved,
            {
                "build": "what a run pays for and another can reuse: exports,"
                " Inductor caches, graph pairs, profiles, lowered programs",
                "planning": "what a run measured: PressureFit requests and"
                " results, and the plans callables run",
            },
            _CACHE_README,
        )
        if self.plan_store is not None:
            self.plan_store.mkdir(parents=True, exist_ok=True)
            _write_guides(
                self.plan_store,
                False,
                {
                    "planning": "PressureFit requests and results, and the"
                    " plans callables run"
                },
                _PLAN_STORE_README,
                artifact_store=self.root,
            )

    def record(
        self,
        *,
        category: str,
        kind: str,
        digest: str | None,
        path: str | Path,
        access: str,
        schema: str | None,
        dependencies: tuple[str, ...] = (),
    ) -> None:
        self._ledger.append(
            PlanningArtifact(
                category,
                kind,
                digest,
                Path(path).expanduser().resolve(),
                access,
                schema,
                dependencies,
            )
        )

    def artifacts(self) -> tuple[PlanningArtifact, ...]:
        return self._ledger.snapshot()

    def diagnostics(self) -> tuple[tuple[str, str], ...]:
        return (
            ("root", str(self.root)),
            ("build", str(self.build)),
            ("build.inductor", str(self.inductor)),
            ("planning", str(self.planning)),
            (
                "plan_store",
                str(self.root if self.plan_store is None else self.plan_store),
            ),
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

        directory = _digest_directory(self.exports, digest)
        artifact_path = directory / "exported_program.pt2"
        manifest_path = directory / "manifest.json"
        if not self.save_plan:
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
            or self.overwrite_plan
        ):
            return False
        manifest = _read_json(manifest_path)
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
        _atomic_json(
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
            self.record(
                category="pytorch",
                kind="export_manifest",
                digest=digest,
                path=manifest_path,
                access="read",
                schema=_EXPORT_SCHEMA,
            )
        self.record(
            category="pytorch",
            kind="exported_program",
            digest=digest,
            path=artifact_path,
            access=artifact_access,
            schema=_EXPORT_SCHEMA,
        )
        if artifact_access == "matched":
            return
        self.record(
            category="pytorch",
            kind="export_manifest",
            digest=digest,
            path=manifest_path,
            access="write",
            schema=_EXPORT_SCHEMA,
        )

    def archive_program(self, program: Program) -> Path:
        """Persist the exact canonical Program supplied to PressureFit."""

        path = (
            _digest_directory(self.pressurefit_programs, program.digest)
            / "program.json"
        )
        if not self.save_plan:
            return path
        encoded = program.to_json()
        operation = "matched" if path.exists() else "write"
        if path.exists() and not self.overwrite_plan:
            try:
                existing = path.read_text()
            except OSError as exc:
                raise ValueError(f"Program cache entry {path} cannot be read") from exc
            if existing != encoded:
                raise ValueError(f"Program cache entry {path} is corrupt")
        else:
            _atomic_text(path, encoded)
            operation = "write"
        self.record(
            category="pressurefit",
            kind="program",
            digest=program.digest,
            path=path,
            access=operation,
            schema=PROGRAM_SCHEMA,
        )
        return path

    def archive_pressurefit_request(
        self,
        value: Mapping[str, object],
    ) -> tuple[str, Path]:
        """Persist one complete, framework-free PressureFit call boundary."""

        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        path = _digest_directory(self.pressurefit_requests, digest) / "request.json"
        if not self.save_plan:
            return digest, path
        operation = "matched" if path.exists() else "write"
        if path.exists() and not self.overwrite_plan:
            try:
                existing = path.read_text()
            except OSError as exc:
                raise ValueError(
                    f"PressureFit request artifact {path} cannot be read"
                ) from exc
            if existing != encoded:
                raise ValueError(f"PressureFit request artifact {path} is corrupt")
        else:
            _atomic_text(path, encoded)
            operation = "write"
        self.record(
            category="pressurefit",
            kind="request",
            digest=digest,
            path=path,
            access=operation,
            schema=artifact_schema("pressurefit_request"),
            dependencies=(str(value["program_digest"]),),
        )
        return digest, path

    def archive_plan(
        self,
        *,
        model_label: str,
        capture_identity: str,
        execution_plan: ExecutionPlan,
        initial_execution_plan: ExecutionPlan | None,
        manifest: Mapping[str, object],
    ) -> Path:
        """Write the resolved plan and a readable lineage manifest."""

        directory = (
            self.plans
            / _safe_label(model_label)
            / capture_identity[:16]
            / execution_plan.digest[:16]
        )
        if not self.save_plan:
            return directory / "manifest.json"
        plan_path = directory / "execution_plan.json"
        _atomic_text(plan_path, execution_plan.to_json())
        self.record(
            category="plans",
            kind="execution_plan",
            digest=execution_plan.digest,
            path=plan_path,
            access="write",
            schema=artifact_schema("execution_plan"),
            dependencies=(execution_plan.program.digest,),
        )
        initial_path: Path | None = None
        if initial_execution_plan is not None:
            initial_path = directory / "initial_execution_plan.json"
            _atomic_text(initial_path, initial_execution_plan.to_json())
            self.record(
                category="plans",
                kind="initial_execution_plan",
                digest=initial_execution_plan.digest,
                path=initial_path,
                access="write",
                schema=artifact_schema("execution_plan"),
                dependencies=(initial_execution_plan.program.digest,),
            )
        manifest_path = directory / "manifest.json"
        _atomic_json(
            manifest_path,
            {
                "schema": _PLAN_MANIFEST_SCHEMA,
                "model": model_label,
                "capture_identity": capture_identity,
                "execution_plan_digest": execution_plan.digest,
                "execution_plan": plan_path.name,
                "initial_execution_plan": (
                    None if initial_path is None else initial_path.name
                ),
                **dict(manifest),
            },
        )
        self.record(
            category="plans",
            kind="plan_manifest",
            digest=execution_plan.digest,
            path=manifest_path,
            access="write",
            schema=_PLAN_MANIFEST_SCHEMA,
            dependencies=(execution_plan.program.digest,),
        )
        return manifest_path


def _clear_pytorch_compiler_caches() -> None:
    """Clear process-local compiler state at an isolated cache boundary."""

    # ShadowSpill's PyTorch frontend is version-pinned.  This private helper is
    # deliberately confined here; it prevents an earlier plan in this process
    # from violating force_fresh's no-read contract.
    from torch._inductor.utils import clear_caches

    clear_caches()


def _publish_cache_tree(source: Path, destination: Path, *, overwrite: bool) -> None:
    """Publish a fresh, write-enabled Inductor cache without replaying old data."""

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
                    "fresh PyTorch compiler artifact conflicts with an existing "
                    "entry; use overwrite_plan=True or a new "
                    f"implementation_revision: {destination_path}"
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


#: Where each directory of the layout before the build/planning split went.
_LEGACY_MOVES = (
    ("pytorch/exports", "build/exports"),
    ("pytorch/inductor", "build/inductor"),
    ("graphpairs", "build/graphpairs"),
    ("profiling", "build/profiling"),
    ("pressurefit/programs", "build/programs"),
    ("pressurefit/requests", "planning/requests"),
    ("pressurefit/selections", "planning/results"),
    ("plans", "planning/plans"),
)


def _migrate_layout(root: Path) -> bool:
    """Move a store laid out before the split into place; True if anything moved.

    Renames, not copies: a store is content-addressed, so a directory that
    already exists at the destination is merged entry by entry and an entry
    already there is left alone.
    """

    moved = False
    for old, new in _LEGACY_MOVES:
        moved = _move_tree(root / old, root / new) or moved
    for stale in ("pytorch", "pressurefit"):
        with suppress(OSError):
            (root / stale).rmdir()
    return moved


def _move_tree(source: Path, target: Path) -> bool:
    if not source.is_dir():
        return False
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
        return True
    for child in source.iterdir():
        destination = target / child.name
        if child.is_dir() and destination.is_dir():
            _move_tree(child, destination)
        elif not destination.exists():
            child.rename(destination)
    with suppress(OSError):
        source.rmdir()
    return True


def _write_guides(
    root: Path,
    replace: bool,
    directories: Mapping[str, str],
    readme: str,
    *,
    artifact_store: Path | None = None,
) -> None:
    layout = root / "layout.json"
    if replace or not layout.exists():
        value: dict[str, object] = {
            "schema": _LAYOUT_SCHEMA,
            "directories": directories,
        }
        if artifact_store is not None:
            value["artifact_store"] = str(artifact_store)
        _atomic_json(layout, value)
    guide = root / "README.md"
    if replace or not guide.exists():
        _atomic_text(guide, readme)


def _store_root(value: Any, name: str) -> Path:
    """The versioned root under a directory a caller named."""

    try:
        root = Path(value).expanduser().resolve()
    except TypeError as exc:
        raise TypeError(f"{name} must be path-like") from exc
    if root.exists() and not root.is_dir():
        raise ValueError(f"{name} must name a directory")
    return root / f"v{ARTIFACT_VERSION}"


def digest_directory(root: Path, digest: str) -> Path:
    """Where one content-addressed entry lives, under any store root.

    Every content-addressed artifact follows one shape,
    ``<kind>/<first two of digest>/<digest>/<document>``: a directory named
    for the key, sharded so no directory grows unbounded, holding one file
    per document. A kind that needs a second file later adds it beside the
    first instead of inventing a path.
    """

    if len(digest) != 64:
        raise ValueError("content-addressed cache key must be SHA-256")
    return root / digest[:2] / digest


_digest_directory = digest_directory


def _safe_label(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return result or "model"


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cache manifest {path} cannot be read") from exc
    if not isinstance(value, dict):
        raise ValueError(f"cache manifest {path} is not an object")
    return value


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as output:
            output.write(value)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        with suppress(FileNotFoundError):
            os.unlink(temporary)


_CACHE_README = """# ShadowSpill planning cache

This directory is both a content-addressed cache and a planning evidence store.
Digests determine identity; readable model names under `plans/` are indexes only.
Its name is the artifact version every file in it carries: a ShadowSpill
update that changes any stored structure writes a fresh `v<N>` tree beside
this one and replans.

`build/` is what a run pays for and another run can reuse:

- `build/exports/`: normalized Export archives and manifests.
- `build/inductor/`: files managed internally by PyTorch Inductor.
- `build/graphpairs/`: structural AOT graph pairs.
- `build/profiling/`: hardware/compiler-specific layouts and task measurements.
- `build/programs/`: exact canonical Programs supplied to PressureFit.

`planning/` is what a run measured:

- `planning/requests/`: what each PressureFit search was asked for.
- `planning/results/`: PressureFit's answer, the selected resolution and
  memory schedule with the search diagnostics.
- `planning/plans/`: one readable manifest and ExecutionPlan per planning call.

A planning call given a plan store keeps `planning/` there instead, so
several runs can share this store and each own its plans.

Every returned `PlanReport` records the absolute path and access disposition of
the artifacts touched by that call.  Do not edit content-addressed entries.
"""

_PLAN_STORE_README = """# ShadowSpill plan store

The plans one run searched, kept apart from the artifact store they were
searched over (named in `layout.json`) so that store can be shared.

- `planning/requests/`: what each PressureFit search was asked for.
- `planning/results/`: PressureFit's answer, the selected resolution and
  memory schedule with the search diagnostics.
- `planning/plans/`: one readable manifest and ExecutionPlan per planning call.

Digests determine identity; do not edit content-addressed entries.
"""


__all__ = ["ArtifactStore", "PlanningArtifact", "digest_directory"]
