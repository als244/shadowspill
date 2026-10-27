"""The artifact store: two trees of content-addressed artifacts under one root."""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, get_args

from shadowspill._files import (
    atomic_json as atomic_json,
)
from shadowspill._files import (
    atomic_text as atomic_text,
)
from shadowspill._files import (
    atomic_write as atomic_write,
)
from shadowspill.ir import ShadowSpillProgram
from shadowspill.ir.program import PROGRAM_SCHEMA
from shadowspill.schema import ARTIFACT_VERSION, artifact_schema

from .policy import STORE_MODES, StoreMode, StorePolicy

_LAYOUT_SCHEMA = artifact_schema("artifact_store")
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
    """Optional content-addressed caches for capture and planning.

    ``build`` holds graph pairs and optimizer captures;
    ``planning`` holds costed IR inputs, search requests and results. Either
    tree can use its own root and read/write policy. Saved public phase
    artifacts are independent of these directories.
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
    def graph_pairs(self) -> Path:
        return self.build / "graph_pairs"

    @property
    def optimizer_captures(self) -> Path:
        return self.build / "optimizers"

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
        for directory in (self.build, self.planning):
            directory.mkdir(parents=True, exist_ok=True)
        _write_guides(
            self.root,
            False,
            {
                "build": "graph pairs and optimizer captures",
                "planning": "costed IR inputs, search requests and results",
            },
            _CACHE_README,
        )
        if self.plan_store is not None:
            self.plan_store.mkdir(parents=True, exist_ok=True)
            _write_guides(
                self.plan_store,
                False,
                {"planning": "costed IR inputs, search requests and results"},
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


def read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cache manifest {path} cannot be read") from exc
    if not isinstance(value, dict):
        raise ValueError(f"cache manifest {path} is not an object")
    return value


_CACHE_README = """# ShadowSpill internal cache

Optional content-addressed entries under a versioned root:

- `build/graph_pairs/`: structural AOT graph pairs.
- `build/optimizers/`: captured optimizer updates.
- `planning/programs/`: costed IR inputs.
- `planning/requests/`: search inputs and settings.
- `planning/results/`: selected schedules, diagnostics and certificates.

Compiler caches remain external and use their compiler's cache settings.
Public phase artifacts use explicit save/load and do not depend on this tree.
Do not edit content-addressed entries.
"""

_PLAN_STORE_README = """# ShadowSpill planner cache

This separate root stores costed IR inputs under `planning/programs/`, search
requests under `planning/requests/`, and results under `planning/results/`.
The shared artifact root is recorded in `layout.json`.

Public Plan archives are saved explicitly. Do not edit cache entries.
"""


__all__ = [
    "STORE_MODES",
    "ArtifactStore",
    "PlanningArtifact",
    "StoreMode",
    "StorePolicy",
    "digest_directory",
]
