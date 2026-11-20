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
    "certified",
    "verdict",
    "matched",
    "read",
    "write",
}


class ArtifactRecorder(Protocol):
    """A function a store calls for each artifact it reads or writes.

    It takes the keyword arguments of `ArtifactStore.record`, which is one
    such function.
    """

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
    """One file or directory a store read or wrote for a call.

    `ArtifactStore.artifacts` returns these records.

    Attributes
    ----------
    category : str
        What touched it, such as ``"search"``, ``"graph_pairs"`` or
        ``"profiling"``.
    kind : str
        What it is within its category, such as ``"program"`` or
        ``"request"``.
    digest : str or None
        The SHA-256 key it is filed under, in hexadecimal; None for an entry
        with no key.
    path : pathlib.Path
        Where it is, as an absolute path.
    access : str
        How it was touched: ``"read"``, ``"write"``, ``"matched"`` (an
        existing entry equal to what would have been written), ``"improved"``
        (a write replacing a stored plan the answer beat), ``"certified"`` (a
        certificate written beside a plan) or ``"verdict"`` (a refusal
        recorded for the key).
    schema : str or None
        The schema label of its contents, if it has one.
    dependencies : tuple of str
        Digests of the artifacts it was derived from.
    """

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
    """Content-addressed caches for capture, measurement and planning.

    `ArtifactStore.resolve` returns it. The store is optional: saved phase
    artifacts never depend on it. Its build tree holds graph pairs and optimizer
    captures, which any run may reuse; its planning tree holds archived programs,
    search requests and results, which a run may keep to itself. Each tree has
    its own root and mode. Task and overlap measurements live under ``root``.

    Attributes
    ----------
    root : pathlib.Path
        The versioned root, ``<directory>/v<N>``.
    build : pathlib.Path
        The build tree.
    planning : pathlib.Path
        The planning tree.
    build_store_mode, plan_store_mode : str
        What the run may do with each tree: ``"contribute"``, ``"reuse"``,
        ``"require"`` or ``"refresh"``.
    export_bypass_key : str or None
        The caller's label for the code the build is made from. The store
        checks and keeps it, and files nothing under it.
    plan_store, build_store : pathlib.Path or None
        The versioned root a tree was given apart from ``root``; None when it
        lives under ``root``.
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
        """Resolve the store a caller names, without creating anything.

        Parameters
        ----------
        value : path-like or None
            The directory both trees live under; None for
            ``~/.cache/shadowspill``.
            The store's root is its ``v<N>`` subdirectory.
        build_store : path-like, optional
            A directory of its own for the build tree.
        plan_store : path-like, optional
            A directory of its own for the planning tree.
        build_store_mode : str, default "contribute"
            What the run may do with the build tree: ``"contribute"``,
            ``"reuse"``, ``"require"`` or ``"refresh"``.
        plan_store_mode : str, default "contribute"
            What the run may do with the planning tree, from the same four.
        export_bypass_key : str, optional
            The caller's label for the code the build is made from; surrounding
            whitespace is stripped.

        Returns
        -------
        ArtifactStore
            The store.

        Raises
        ------
        ValueError
            If a mode is not one of the four, ``export_bypass_key`` is blank, or a
            directory names something that is not a directory.
        TypeError
            If ``export_bypass_key`` is not a str, or a directory is not path-like.
        """
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
        """Where the build tree keeps graph pairs."""
        return self.build / "graph_pairs"

    @property
    def optimizer_captures(self) -> Path:
        """Where the build tree keeps optimizer captures."""
        return self.build / "optimizers"

    @property
    def task_profiles(self) -> Path:
        """Where task measurements are kept, under ``root``."""
        return self.root / "profiling" / "tasks"

    @property
    def overlap_profiles(self) -> Path:
        """Where overlap measurements are kept, under ``root``."""
        return self.root / "profiling" / "overlap"

    @property
    def programs_archive(self) -> Path:
        """Where the planning tree keeps the programs it was asked to plan.

        A plan's lineage points at this immutable copy.
        """
        # In the planning tree, not the build tree. A planning call archives
        # the program it was given so a plan's lineage points at an immutable
        # copy of what was planned -- that copy is evidence about the planning,
        # and planning writes nothing under `build` by design.
        return self.planning / "programs"

    @property
    def plan_requests(self) -> Path:
        """Where the planning tree keeps search requests."""
        return self.planning / "requests"

    @property
    def plan_selections(self) -> Path:
        """Where the planning tree keeps search results."""
        return self.planning / "results"

    @property
    def build_policy(self) -> StorePolicy:
        """What this run may do with the build tree."""

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
        """Create the store's directories and the guides that describe them.

        Creates the root, both trees and the profiling directory, and writes a
        ``layout.json`` and a ``README.md`` at the root; a separate plan store
        gets its own guides, naming the root. Existing guides are kept.
        """

        self.root.mkdir(parents=True, exist_ok=True)
        for directory in (self.build, self.root / "profiling", self.planning):
            directory.mkdir(parents=True, exist_ok=True)
        _write_guides(
            self.root,
            False,
            {
                "build": "graph pairs and optimizer captures",
                "profiling": "finalized task measurements",
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
        """Record one artifact this store touched.

        The record joins `artifacts` unless an equal one is already there. Safe to
        call from several threads.

        Parameters
        ----------
        category : str
            What touched it, such as ``"search"``.
        kind : str
            What it is within its category, such as ``"program"``.
        digest : str or None
            The SHA-256 key it is filed under, in hexadecimal, or None.
        path : str or pathlib.Path
            Where it is; recorded as an absolute path.
        access : str
            How it was touched, as in `PlanningArtifact.access`.
        schema : str or None
            The schema label of its contents, if it has one.
        dependencies : tuple of str, default ()
            Digests of the artifacts it was derived from.

        Raises
        ------
        ValueError
            If ``category`` or ``kind`` is empty, ``digest`` is not 64 characters,
            or ``access`` is unknown.
        """
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
        """Return the artifacts this store touched, in the order first recorded.

        Returns
        -------
        tuple of PlanningArtifact
            One record each.
        """
        return self._ledger.snapshot()

    def diagnostics(self) -> tuple[tuple[str, str], ...]:
        """Return the store's roots, for a report.

        Returns
        -------
        tuple of (str, str)
            ``("root", ...)``, ``("build", ...)``, ``("planning", ...)`` and
            ``("plan_store", ...)``, each with its path; the plan store is the root
            when it has none of its own.
        """
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
        """Archive the exact program a search was given, in the planning tree.

        Nothing is written when the planning policy forbids writing. An existing
        entry is compared rather than rewritten, unless the policy overwrites.

        Parameters
        ----------
        program : ShadowSpillProgram
            The program.

        Returns
        -------
        pathlib.Path
            Where the program's canonical JSON is, under its digest.

        Raises
        ------
        ValueError
            If an existing entry cannot be read or differs from the program.
        """

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
        """Archive one search request in the planning tree, keyed by its contents.

        Nothing is written when the planning policy forbids writing. An existing
        entry is compared rather than rewritten, unless the policy overwrites.

        Parameters
        ----------
        value : mapping
            The request, as JSON values, with the ``"program_digest"`` of the
            program it plans.

        Returns
        -------
        digest : str
            SHA-256 of the request's compact sorted JSON, in hexadecimal.
        path : pathlib.Path
            Where the request is.

        Raises
        ------
        ValueError
            If an existing entry cannot be read or differs from the request.
        """

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
    """Return the directory one content-addressed entry lives in.

    Every entry is ``<root>/<first two characters of digest>/<digest>/``,
    holding one file per document.

    Parameters
    ----------
    root : pathlib.Path
        The directory of one kind of entry.
    digest : str
        The entry's SHA-256 key, in hexadecimal.

    Returns
    -------
    pathlib.Path
        The entry's directory.

    Raises
    ------
    ValueError
        If ``digest`` is not 64 characters.
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
- `profiling/tasks/`: finalized task timing and allocation evidence.
- `profiling/overlap/`: compatible paired task/collective evidence.
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
