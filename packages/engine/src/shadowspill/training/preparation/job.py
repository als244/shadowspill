"""Saved resolved preparation jobs for the existing fresh-phase launcher."""

from __future__ import annotations

import fcntl
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from shadowspill.artifacts.archive import Artifact
from shadowspill.orchestration.launch.phases import PhaseInvocation, PhaseResult
from shadowspill.training.definitions import Registry, load_registry
from shadowspill_client._records import names
from shadowspill_client.protocol.training import ResolvedTraining

from .bundle import TrainingBundle
from .layouts import prepare_layouts
from .phases import run_phases
from .setup import resolve_training
from .templates import TemplateInventory, capture_inventory

JOB_NAME = "preparation.ssp"
BUNDLE_NAME = "training.ssp"


@contextmanager
def _job_lock(root: Path) -> Iterator[None]:
    # Keep the inode after release: unlinking it could let another caller lock
    # a new inode while an existing waiter still owns the previous one.
    with (root / ".preparation.lock").open("ab") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("preparation job is already running") from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


@dataclass(frozen=True)
class PreparationJob(Artifact):
    """Server-owned configuration; package names never come from compute requests."""

    artifact_kind: ClassVar[str] = "preparation_job"
    metadata_limit_bytes: ClassVar[int | None] = 16 << 20
    payload_limit_bytes: ClassVar[int | None] = 0
    packages: tuple[str, ...]
    config: ResolvedTraining

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "packages", names(self.packages, "definition packages")
        )
        if not isinstance(self.config, ResolvedTraining):
            raise TypeError("preparation job requires a resolved training setup")

    def resolve(self) -> tuple[Registry, TemplateInventory]:
        registry = load_registry(self.packages)
        setup = resolve_training(registry, config=self.config)
        return registry, capture_inventory(prepare_layouts(setup))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "packages": list(self.packages),
            "config": self.config.to_dict(),
        }

    @classmethod
    def from_dict(
        cls, value: dict[str, Any], *, payloads: Mapping[str, bytes] | None = None
    ) -> PreparationJob:
        if (
            payloads
            or set(value) != {"version", "packages", "config"}
            or type(value["version"]) is not int
            or value["version"] != 1
        ):
            raise ValueError("unsupported preparation job")
        return cls(value["packages"], ResolvedTraining.from_dict(value["config"]))


def prepare_job(
    directory: str | Path,
    *,
    packages: Sequence[str],
    config: ResolvedTraining,
    environment: Mapping[str, str] | None = None,
    launch: Callable[[PhaseInvocation], PhaseResult] | None = None,
) -> TrainingBundle:
    """Prepare on assigned workers, or load this job's completed bundle.

    The caller owns allocation admission, launch environment and the directory.
    This launches no training loop or persistent Executor and initializes no weights.
    Loading verifies archive integrity and the exact job configuration. The service
    calls ``bundle.recheck`` once at execution admission, for fresh and saved Plans.
    """
    with _run_job(
        directory,
        packages=packages,
        config=config,
        environment=environment,
        launch=launch,
        phases=("build", "profile", "communicate", "plan"),
    ) as (root, _):
        bundle = TrainingBundle.load(root / BUNDLE_NAME)
        if bundle.config != config:
            raise ValueError("completed bundle differs from the saved preparation job")
        return bundle


def measure_job(
    directory: str | Path,
    *,
    packages: Sequence[str],
    config: ResolvedTraining,
    environment: Mapping[str, str] | None = None,
    launch: Callable[[PhaseInvocation], PhaseResult] | None = None,
) -> Path:
    """Run the same preparation through measurement, without planning or workers.

    The directory retains every requested build case and its measured artifacts
    or recoverable rejection. Explicit invocation may reuse exact build/profile
    files; communication measurement runs on the assigned target workers.
    """
    with _run_job(
        directory,
        packages=packages,
        config=config,
        environment=environment,
        launch=launch,
        phases=("build", "profile", "communicate"),
    ) as (root, _):
        return root


@contextmanager
def _run_job(
    directory: str | Path,
    *,
    packages: Sequence[str],
    config: ResolvedTraining,
    environment: Mapping[str, str] | None,
    launch: Callable[[PhaseInvocation], PhaseResult] | None,
    phases: tuple[str, ...],
) -> Iterator[tuple[Path, Registry]]:
    from .measurement_reuse import require_staged_reuse

    root = Path(directory).resolve()
    require_staged_reuse(root, config)
    requested = PreparationJob(tuple(packages), config)
    registry, inventory = requested.resolve()
    created = False
    try:
        root.mkdir(parents=True)
        created = True
    except FileExistsError:
        if not (root / JOB_NAME).is_file():
            raise ValueError(
                "existing directory has no matching preparation job"
            ) from None
    with _job_lock(root):
        if created:
            requested.save(root / JOB_NAME)
        elif PreparationJob.load(root / JOB_NAME) != requested:
            raise ValueError("existing preparation job has different packages or setup")
        if not (root / BUNDLE_NAME).exists():
            ranks = inventory.layouts.setup.context.parallelism.world_size
            run_phases(
                "shadowspill.training.preparation",
                root,
                "all",
                ranks=ranks,
                preparation_ranks=ranks,
                phases=phases,
                environment=environment,
                launch=launch,
            )
        yield root, registry


def load_job(directory: str | Path) -> tuple[Path, Registry, TemplateInventory]:
    from .measurement_reuse import require_staged_reuse

    root = Path(directory)
    job = PreparationJob.load(root / JOB_NAME)
    require_staged_reuse(root, job.config)
    registry, inventory = job.resolve()
    return root, registry, inventory
