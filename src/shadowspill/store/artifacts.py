"""The artifact store: two trees of content-addressed artifacts under one root."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, get_args

from shadowspill.ir import ExecutionPlan, ShadowSpillProgram
from shadowspill.ir.program import PROGRAM_SCHEMA
from shadowspill.schema import ARTIFACT_VERSION, artifact_schema

from .policy import STORE_MODES, StoreMode, StorePolicy

_LAYOUT_SCHEMA = artifact_schema("artifact_store")
_PLAN_MANIFEST_SCHEMA = artifact_schema("plan_manifest")
#: How a run touched an artifact. "improved" is a write that replaced a
#: stored plan the answer beat, kept distinct from a first write so the
#: ledger says which plans a run displaced.
_ACCESS_KINDS = {
    "improved",
    "managed",
    "certified",
    "verdict",
    "matched",
    "read",
    "write",
}


class ArtifactRecorder(Protocol):
    """What a store tells about each artifact it reads or writes: the
    signature of `ArtifactStore.record`."""

    def __call__(
        self,
        *,
        category: str,
        kind: str,
        digest: str | None,
        path: str | Path,
        access: str,
        schema: str | None,
        dependencies: tuple[str, ...] = (),
    ) -> None: ...


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
    and another run can reuse: exports, the compiler caches, graph pairs,
    optimizer captures and profiles. ``planning`` holds what a run decided:
    the program it was given,
    the requests put to the planner, its results, and the plans callables run,
    the last a readable index linking one planning call to the immutable
    artifacts behind it. Nothing under ``planning`` is written by a build, and
    nothing under ``build`` by a planning call, which is what lets one build
    store serve many runs that each keep their own plans. Content-addressed
    leaf paths provide identity throughout.

    ``plan_store`` puts the ``planning`` tree under another directory, so
    several runs can share one artifact store and each keep its own plans;
    ``None`` keeps both trees under ``root``.
    """

    root: Path
    build: Path
    planning: Path
    build_store_mode: StoreMode = "contribute"
    plan_store_mode: StoreMode = "contribute"
    export_bypass_key: str | None = None
    plan_store: Path | None = None
    build_store: Path | None = None
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
        build_store: Any | None = None,
        plan_store: Any | None = None,
        build_store_mode: StoreMode = "contribute",
        plan_store_mode: StoreMode = "contribute",
        export_bypass_key: str | None = None,
    ) -> ArtifactStore:
        for name, mode in (
            ("build_store_mode", build_store_mode),
            ("plan_store_mode", plan_store_mode),
        ):
            if mode not in get_args(StoreMode.__value__):
                allowed = ", ".join(get_args(StoreMode.__value__))
                raise ValueError(f"{name} must be one of {allowed}: got {mode!r}")
        if export_bypass_key is not None:
            if not isinstance(export_bypass_key, str):
                raise TypeError("export_bypass_key must be a string or None")
            export_bypass_key = export_bypass_key.strip()
            if not export_bypass_key:
                raise ValueError("export_bypass_key must be non-empty")
        root = (
            _store_root(value, "artifact_store")
            if value is not None
            else (Path.home() / ".cache" / "shadowspill").resolve()
            / f"v{ARTIFACT_VERSION}"
        )
        # Either tree may be rooted somewhere of its own, and a specific root
        # wins over the shared one. The common shape is a build store several
        # runs read and a plan store each run keeps to itself.
        plan_root = (
            None if plan_store is None else _store_root(plan_store, "plan_store")
        )
        build_root = (
            None if build_store is None else _store_root(build_store, "build_store")
        )
        return cls(
            root,
            (root if build_root is None else build_root) / "build",
            (root if plan_root is None else plan_root) / "planning",
            build_store_mode,
            plan_store_mode,
            export_bypass_key,
            plan_store=plan_root,
            build_store=build_root,
        )

    @property
    def exports(self) -> Path:
        return self.build / "exports"

    @property
    def graphpairs(self) -> Path:
        return self.build / "graphpairs"

    @property
    def optimizer_captures(self) -> Path:
        return self.build / "optimizers"

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
    def steps(self) -> Path:
        # Step programs by the identity a build has before any capture, so a
        # build with a bypass key can answer without exporting.
        return self.build / "steps"

    @property
    def programs_archive(self) -> Path:
        # In the planning tree, not the build tree. A planning call archives
        # the program it was given so a plan's lineage points at an immutable
        # copy of what was planned -- that copy is evidence about the planning,
        # and planning writes nothing under `build` by design.
        return self.planning / "programs"

    @property
    def plan_requests(self) -> Path:
        return self.planning / "requests"

    @property
    def plan_selections(self) -> Path:
        return self.planning / "results"

    @property
    def plans(self) -> Path:
        return self.planning / "plans"

    @property
    def build_policy(self) -> StorePolicy:
        """What this run may do with the build tree.

        Held apart from the planning tree because the two are shared for
        different reasons. A build artifact is what a run *paid for* and any
        run may reuse; a plan is what a run *decided*, and two runs comparing
        planners must not read each other's.
        """

        return StorePolicy.for_mode(self.build_store_mode)

    @property
    def plan_policy(self) -> StorePolicy:
        """What this run may do with the planning tree."""

        return StorePolicy.for_mode(self.plan_store_mode)

    @property
    def build_reads_enabled(self) -> bool:
        """Whether this run uses build entries the store already holds."""

        return self.build_policy.read_enabled

    @property
    def build_writes_enabled(self) -> bool:
        """Whether this run may add to the build tree."""

        return self.build_policy.write_enabled

    def initialize(self) -> None:
        """Create the stable top-level layout and its human guide."""

        self.root.mkdir(parents=True, exist_ok=True)
        for directory in (self.build, self.profiling, self.planning):
            directory.mkdir(parents=True, exist_ok=True)
        _write_guides(
            self.root,
            False,
            {
                "build": "what a run pays for and another can reuse: exports,"
                " the compiler caches, graph pairs, optimizer captures, profiles",
                "planning": "what a run decided: the programs it was given,"
                " the requests put to the planner, its results, and the plans"
                " callables run",
            },
            _CACHE_README,
        )
        if self.plan_store is not None:
            self.plan_store.mkdir(parents=True, exist_ok=True)
            _write_guides(
                self.plan_store,
                False,
                {
                    "planning": "search requests and results, and the"
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
            ("planning", str(self.planning)),
            (
                "plan_store",
                str(self.root if self.plan_store is None else self.plan_store),
            ),
        )

    def archive_program(self, program: ShadowSpillProgram) -> Path:
        """Persist the exact canonical ShadowSpillProgram the search was given."""

        path = digest_directory(self.programs_archive, program.digest) / "program.json"
        if not self.plan_policy.write_enabled:
            return path
        encoded = program.to_json()
        operation = "matched" if path.exists() else "write"
        if path.exists() and not self.plan_policy.overwrite:
            try:
                existing = path.read_text()
            except OSError as exc:
                raise ValueError(f"program store entry {path} cannot be read") from exc
            if existing != encoded:
                raise ValueError(f"program cache entry {path} is corrupt")
        else:
            atomic_text(path, encoded)
            operation = "write"
        self.record(
            category="search",
            kind="program",
            digest=program.digest,
            path=path,
            access=operation,
            schema=PROGRAM_SCHEMA,
        )
        return path

    def archive_plan_request(
        self,
        value: Mapping[str, object],
    ) -> tuple[str, Path]:
        """Persist one complete, framework-free search call boundary."""

        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        path = digest_directory(self.plan_requests, digest) / "request.json"
        if not self.plan_policy.write_enabled:
            return digest, path
        operation = "matched" if path.exists() else "write"
        if path.exists() and not self.plan_policy.overwrite:
            try:
                existing = path.read_text()
            except OSError as exc:
                raise ValueError(
                    f"search request artifact {path} cannot be read"
                ) from exc
            if existing != encoded:
                raise ValueError(f"search request artifact {path} is corrupt")
        else:
            atomic_text(path, encoded)
            operation = "write"
        self.record(
            category="search",
            kind="request",
            digest=digest,
            path=path,
            access=operation,
            schema=artifact_schema("plan_request"),
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
            / safe_label(model_label)
            / capture_identity[:16]
            / execution_plan.digest[:16]
        )
        if not self.plan_policy.write_enabled:
            return directory / "manifest.json"
        plan_path = directory / "execution_plan.json"
        atomic_text(plan_path, execution_plan.to_json())
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
            atomic_text(initial_path, initial_execution_plan.to_json())
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
        atomic_json(
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
        atomic_json(layout, value)
    guide = root / "README.md"
    if replace or not guide.exists():
        atomic_text(guide, readme)


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


def safe_label(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return result or "model"


def read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cache manifest {path} cannot be read") from exc
    if not isinstance(value, dict):
        raise ValueError(f"cache manifest {path} is not an object")
    return value


def atomic_json(path: Path, value: Mapping[str, object]) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_text(path: Path, value: str) -> None:
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
- `build/inductor/`: files managed internally by the framework's compiler.
- `build/graphpairs/`: structural AOT graph pairs.
- `build/optimizers/`: traced recurrent optimizer updates, keyed by the
  optimizer, the objects it binds, its hyperparameters and its stage split.
- `build/profiling/`: hardware/compiler-specific layouts and task measurements.

`planning/` is what a run decided:

- `planning/programs/`: the exact canonical ShadowSpillProgram each planning call was
  given, so a plan's lineage points at an immutable copy of what was planned.
- `planning/requests/`: what each search was asked for.
- `planning/results/`: the planner's answer, the selected resolution and
  memory schedule with the search diagnostics.
- `planning/plans/`: one readable manifest and ExecutionPlan per planning call.

The two trees are written by different halves of the work and never by each
other: a build contributes nothing under `planning/`, and a planning call
nothing under `build/`. That is what lets one build store serve many runs
while each keeps its own plans -- point `--build-store` at the shared one and
`--plan-store` at your own, or give a single `--artifact-store` and get both
under it.

Every returned `PlanReport` records the absolute path and access disposition of
the artifacts touched by that call.  Do not edit content-addressed entries.
"""

_PLAN_STORE_README = """# ShadowSpill plan store

The plans one run searched, kept apart from the artifact store they were
searched over (named in `layout.json`) so that store can be shared.

- `planning/requests/`: what each search was asked for.
- `planning/results/`: the search's answer, the selected resolution and
  memory schedule with the search diagnostics.
- `planning/plans/`: one readable manifest and ExecutionPlan per planning call.

Digests determine identity; do not edit content-addressed entries.
"""


__all__ = [
    "STORE_MODES",
    "ArtifactStore",
    "PlanningArtifact",
    "StoreMode",
    "StorePolicy",
    "digest_directory",
]
