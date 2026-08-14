"""Resolve compiler storage manifests through the profiling artifact cache."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.compilation.inductor import ExecutableTaskManifest
from shadowspill.pytorch.compilation.layout import reconcile_compiled_task_layout
from shadowspill.pytorch.contracts import CaptureError, ProfilingError

from .manifest_repository import CompiledManifestRepository
from .records import ProfileEnvironment, ProfileKey, TaskMeasurement
from .repository import ProfileRepository
from .runner import ProfilableArtifact


class ManifestCompiler(Protocol):
    """Minimal compiler capability needed to hydrate missing manifests."""

    def prepare_manifests(
        self,
        artifacts: Sequence[ProfilableArtifact],
        *,
        progress: Callable[[int, int, str, str], None] | None = None,
    ) -> dict[str, ExecutableTaskManifest]: ...


@dataclass(frozen=True, slots=True)
class ResolvedTaskManifests:
    """Structural manifests plus cache hit/miss evidence."""

    manifests: dict[str, ExecutableTaskManifest]
    cache_hits: int
    cache_misses: int


def validate_compiled_profile(
    artifact: ProfilableArtifact,
    measurement: TaskMeasurement,
    manifests: dict[str, ExecutableTaskManifest],
) -> None:
    """Reject a physical profile that disagrees with its compiler ABI."""

    if not isinstance(artifact, GraphArtifact):
        return
    try:
        manifest = manifests[artifact.compatibility_digest]
    except KeyError as exc:
        raise ProfilingError(
            "compiled task manifest is missing during profile validation",
            structural_abi=artifact.compatibility_digest,
            task_kind=artifact.kind,
            operators=tuple(artifact.operator_targets),
        ) from exc
    try:
        reconcile_compiled_task_layout(
            manifest.storage_contract,
            measurement,
            root_allocations=manifest.root_allocations,
        )
    except CaptureError as exc:
        raise ProfilingError(
            "compiled task profile disagrees with structural ABI "
            f"{artifact.compatibility_digest}: {exc}",
            structural_abi=artifact.compatibility_digest,
            task_kind=artifact.kind,
            operators=tuple(artifact.operator_targets),
        ) from exc


def resolve_task_manifests(
    artifacts: Sequence[ProfilableArtifact],
    *,
    environment: ProfileEnvironment,
    profile_cache: ProfileRepository,
    compiler: ManifestCompiler,
    progress: Callable[[int, int, str, str], None] | None = None,
) -> ResolvedTaskManifests:
    """Load storage ABIs and compile only missing profile sidecars."""

    unique = _unique_graph_artifacts(artifacts)
    sidecars = _manifest_cache(profile_cache)
    manifests, missing = _read_manifests(
        unique,
        environment,
        sidecars,
        progress=progress,
    )
    _compile_missing_manifests(
        missing,
        manifests,
        environment,
        sidecars,
        compiler,
    )
    return ResolvedTaskManifests(
        manifests=manifests,
        cache_hits=len(unique) - len(missing),
        cache_misses=len(missing),
    )


def _unique_graph_artifacts(
    artifacts: Sequence[ProfilableArtifact],
) -> dict[str, GraphArtifact]:
    return {
        artifact.compatibility_digest: artifact
        for artifact in artifacts
        if isinstance(artifact, GraphArtifact)
    }


def _manifest_cache(
    profile_cache: ProfileRepository,
) -> CompiledManifestRepository:
    return CompiledManifestRepository(
        profile_cache.compiled_manifest_root,
        read_enabled=profile_cache.read_enabled,
        write_enabled=profile_cache.write_enabled,
        overwrite=profile_cache.overwrite,
        artifact_recorder=profile_cache.artifact_recorder,
    )


def _read_manifests(
    artifacts: dict[str, GraphArtifact],
    environment: ProfileEnvironment,
    cache: CompiledManifestRepository,
    *,
    progress: Callable[[int, int, str, str], None] | None,
) -> tuple[dict[str, ExecutableTaskManifest], list[GraphArtifact]]:
    manifests: dict[str, ExecutableTaskManifest] = {}
    missing: list[GraphArtifact] = []
    for index, digest in enumerate(sorted(artifacts), start=1):
        artifact = artifacts[digest]
        manifest = cache.read(
            ProfileKey(digest, environment),
            semantic_contract=artifact.storage_contract,
        )
        if manifest is None:
            missing.append(artifact)
            state = "cache-miss"
        else:
            manifests[digest] = manifest
            state = "cache-hit"
        if progress is not None:
            progress(index, len(artifacts), state, digest)
    return manifests, missing


def _compile_missing_manifests(
    missing: Sequence[GraphArtifact],
    manifests: dict[str, ExecutableTaskManifest],
    environment: ProfileEnvironment,
    cache: CompiledManifestRepository,
    compiler: ManifestCompiler,
) -> None:
    hydrated = compiler.prepare_manifests(missing)
    for artifact in missing:
        digest = artifact.compatibility_digest
        manifest = hydrated[digest]
        manifests[digest] = manifest
        cache.write(ProfileKey(digest, environment), manifest)


__all__ = [
    "ManifestCompiler",
    "ResolvedTaskManifests",
    "resolve_task_manifests",
    "validate_compiled_profile",
]
