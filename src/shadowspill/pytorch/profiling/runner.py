"""Profile each unique structural task contract and scatter its measurement."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace
from functools import partial
from typing import Protocol

from shadowspill.errors import CaptureError, ProfilingError
from shadowspill.profiling.store import ProfileStore
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.distributed import current as distributed_preparation
from shadowspill.pytorch.distributed._profiling import cases as distributed_cases
from shadowspill.pytorch.distributed._profiling import phase
from shadowspill.task.profiles import (
    ProfileEnvironment,
    ProfileKey,
    ProfilingResult,
    TaskMeasurement,
)
from shadowspill.task.profiling import ProfilingOptions


class ProfilableArtifact(Protocol):
    """Minimal identity shared by compiled and bounded eager tasks."""

    @property
    def compatibility_digest(self) -> str: ...


def profile_unique_artifacts(
    artifacts: Iterable[ProfilableArtifact],
    *,
    environment: ProfileEnvironment,
    measure: Callable[[ProfilableArtifact], TaskMeasurement],
    cache: ProfileStore,
    validate: Callable[[ProfilableArtifact, TaskMeasurement], None] | None = None,
    progress: Callable[[int, int, str, str], None] | None = None,
    profiling_metadata_digests: Sequence[str | None] | None = None,
    allocation_probe_seeds: int = 1,
    allocation_probe_repetitions: int = 2,
    profiling_options: ProfilingOptions | None = None,
) -> ProfilingResult:
    """Measure each structural key once and scatter it to every occurrence."""

    sequence = tuple(artifacts)
    metadata = _aligned_metadata(sequence, profiling_metadata_digests)
    (
        keys,
        positions,
        representatives,
        position_keys,
    ) = _index_profile_keys(
        sequence,
        metadata,
        environment,
        allocation_probe_seeds=allocation_probe_seeds,
        allocation_probe_repetitions=allocation_probe_repetitions,
        profiling_options=profiling_options or ProfilingOptions(),
    )
    if distributed_preparation() is not None:
        return _profile_distributed(
            sequence,
            metadata,
            keys,
            position_keys,
            measure,
            cache,
            validate,
            progress,
            allocation_probe_seeds,
            allocation_probe_repetitions,
        )
    measurements, hits, misses = _measure_unique_keys(
        keys,
        positions,
        representatives,
        measure,
        cache,
        validate,
        progress,
    )
    return _build_profiling_result(
        metadata,
        keys,
        positions,
        position_keys,
        measurements,
        hits,
        misses,
        allocation_probe_seeds,
        allocation_probe_repetitions,
    )


def _profile_distributed(
    sequence: tuple[ProfilableArtifact, ...],
    metadata: tuple[str | None, ...],
    local_keys: dict[str, ProfileKey],
    local_position_keys: tuple[str, ...],
    measure: Callable[[ProfilableArtifact], TaskMeasurement],
    cache: ProfileStore,
    validate: Callable[[ProfilableArtifact, TaskMeasurement], None] | None,
    progress: Callable[[int, int, str, str], None] | None,
    seeds: int,
    repetitions: int,
) -> ProfilingResult:
    bound = distributed_preparation()
    assert bound is not None
    scheduled = distributed_cases(sequence, local_position_keys, purpose="measurement")
    # Lowering currently addresses a profile by local contract and declared
    # metadata. Refuse an ambiguous mapping instead of silently using the last
    # remote context's cost for every occurrence of the local contract.
    contexts: dict[str, str] = {}
    for case in scheduled:
        local = local_position_keys[case.position]
        previous = contexts.setdefault(local, case.identity)
        if previous != case.identity:
            raise ValueError(
                "one local task profile has different peer task contexts; "
                "supply distinct profiling_metadata for those occurrences"
            )
    keys: dict[str, ProfileKey] = {}
    positions: dict[str, list[int]] = {}
    measurements: dict[str, TaskMeasurement] = {}
    position_keys = [""] * len(sequence)
    hits = misses = 0
    for index, case in enumerate(scheduled, start=1):
        artifact = sequence[case.position]
        local_key = local_keys[local_position_keys[case.position]]
        context = hashlib.sha256(
            json.dumps([case.identity, bound.control.rank]).encode()
        ).hexdigest()
        key = replace(local_key, profiling_metadata_digest=context)
        keys[key.digest] = key
        positions[key.digest] = list(case.occurrences)
        for position in case.occurrences:
            position_keys[position] = key.digest

        def read(
            key: ProfileKey = key, artifact: ProfilableArtifact = artifact
        ) -> tuple[TaskMeasurement | None, bool]:
            cached = cache.read(key)
            if cached is not None and validate is not None:
                try:
                    validate(artifact, cached)
                except (CaptureError, ProfilingError):
                    return None, True
            return cached, False

        cached, invalid = bound.control.run(f"measurement/{case.identity}/read", read)
        complete = all(
            bound.control.exchange(
                f"measurement/{case.identity}/cached", cached is not None
            )
        )
        if complete:
            assert cached is not None
            _report_progress(progress, index, len(scheduled), "cache-hit", key.digest)
            measurement = cached
            hits += 1
        else:
            _report_progress(progress, index, len(scheduled), "measuring", key.digest)
            with phase(case.identity):
                measured = bound.control.run(
                    f"measurement/{case.identity}/measure",
                    partial(measure, artifact),
                )
            if validate is not None:
                validate(artifact, measured)
            # A valid local record remains authoritative. Its rank still runs
            # every invocation needed by peers with a cache miss.
            measurement = measured if cached is None else cached
            if cached is None:
                cache.write(key, measurement, replace_invalid=invalid)
            misses += 1
        measurements[key.digest] = measurement
    return _build_profiling_result(
        metadata,
        keys,
        positions,
        tuple(position_keys),
        measurements,
        hits,
        misses,
        seeds,
        repetitions,
    )


def _aligned_metadata(
    artifacts: tuple[ProfilableArtifact, ...],
    metadata: Sequence[str | None] | None,
) -> tuple[str | None, ...]:
    if metadata is None:
        values: tuple[str | None, ...] = (None,) * len(artifacts)
    else:
        values = tuple(metadata)
    if len(values) != len(artifacts):
        raise ValueError("profiling metadata must align with task artifacts")
    return values


def _index_profile_keys(
    artifacts: tuple[ProfilableArtifact, ...],
    metadata: tuple[str | None, ...],
    environment: ProfileEnvironment,
    *,
    allocation_probe_seeds: int,
    allocation_probe_repetitions: int,
    profiling_options: ProfilingOptions,
) -> tuple[
    dict[str, ProfileKey],
    dict[str, list[int]],
    dict[str, ProfilableArtifact],
    tuple[str, ...],
]:
    keys: dict[str, ProfileKey] = {}
    positions: dict[str, list[int]] = {}
    representatives: dict[str, ProfilableArtifact] = {}
    position_keys: list[str] = []
    for position, (artifact, metadata_digest) in enumerate(
        zip(artifacts, metadata, strict=True)
    ):
        key = ProfileKey(
            artifact.compatibility_digest,
            environment,
            metadata_digest,
            allocation_probe_seeds,
            allocation_probe_repetitions,
            profiling_options=profiling_options,
        )
        keys[key.digest] = key
        position_keys.append(key.digest)
        positions.setdefault(key.digest, []).append(position)
        representatives.setdefault(key.digest, artifact)
    return (
        keys,
        positions,
        representatives,
        tuple(position_keys),
    )


def _measure_unique_keys(
    keys: dict[str, ProfileKey],
    positions: dict[str, list[int]],
    representatives: dict[str, ProfilableArtifact],
    measure: Callable[[ProfilableArtifact], TaskMeasurement],
    cache: ProfileStore,
    validate: Callable[[ProfilableArtifact, TaskMeasurement], None] | None,
    progress: Callable[[int, int, str, str], None] | None,
) -> tuple[dict[str, TaskMeasurement], int, int]:
    results: dict[str, TaskMeasurement] = {}
    hits = 0
    misses = 0
    ordered = sorted(positions)
    for index, digest in enumerate(ordered, start=1):
        measurement, hit = _resolve_measurement(
            keys[digest],
            representatives[digest],
            measure,
            cache,
            validate,
            progress,
            index,
            len(ordered),
        )
        results[digest] = measurement
        hits += int(hit)
        misses += int(not hit)
    return results, hits, misses


def _resolve_measurement(
    key: ProfileKey,
    artifact: ProfilableArtifact,
    measure: Callable[[ProfilableArtifact], TaskMeasurement],
    cache: ProfileStore,
    validate: Callable[[ProfilableArtifact, TaskMeasurement], None] | None,
    progress: Callable[[int, int, str, str], None] | None,
    index: int,
    total: int,
) -> tuple[TaskMeasurement, bool]:
    measurement = cache.read(key)
    replace_invalid = False
    if measurement is not None and validate is not None:
        try:
            validate(artifact, measurement)
        except (CaptureError, ProfilingError):
            _report_progress(progress, index, total, "cache-invalid", key.digest)
            measurement = None
            replace_invalid = True
    if measurement is not None:
        _report_progress(progress, index, total, "cache-hit", key.digest)
        return measurement, True
    _report_progress(progress, index, total, "measuring", key.digest)
    measurement = measure(artifact)
    if validate is not None:
        validate(artifact, measurement)
    cache.write(key, measurement, replace_invalid=replace_invalid)
    return measurement, False


def _report_progress(
    progress: Callable[[int, int, str, str], None] | None,
    index: int,
    total: int,
    state: str,
    digest: str,
) -> None:
    if progress is not None:
        progress(index, total, state, digest)


def _build_profiling_result(
    metadata: tuple[str | None, ...],
    keys: dict[str, ProfileKey],
    positions: dict[str, list[int]],
    position_keys: tuple[str, ...],
    measurements: dict[str, TaskMeasurement],
    hits: int,
    misses: int,
    allocation_probe_seeds: int,
    allocation_probe_repetitions: int,
) -> ProfilingResult:
    by_position: list[TaskMeasurement | None] = [None] * len(position_keys)
    persistent_by_graph: dict[str, int] = {}
    for digest, occurrences in positions.items():
        measurement = measurements[digest]
        graph_digest = keys[digest].graph_digest
        persistent_by_graph[graph_digest] = max(
            persistent_by_graph.get(graph_digest, 0),
            sum(measurement.persistent_extent_bytes),
        )
        for position in occurrences:
            by_position[position] = measurement
    if any(item is None for item in by_position):
        raise AssertionError("profiling result scatter is incomplete")
    return ProfilingResult(
        measurements=tuple(item for item in by_position if item is not None),
        unique_keys=len(positions),
        cache_hits=hits,
        cache_misses=misses,
        fixed_slab_bytes=sum(persistent_by_graph.values()),
        key_digests=position_keys,
        profiling_metadata_digests=metadata,
        allocation_probe_seeds=allocation_probe_seeds,
        allocation_probe_repetitions=allocation_probe_repetitions,
    )


__all__ = ["ProfilableArtifact", "profile_unique_artifacts"]


def unique_graph_artifacts(
    artifacts: Sequence[ProfilableArtifact],
) -> dict[str, GraphArtifact]:
    """The graph artifacts among `artifacts`, one per compatibility digest."""

    return {
        artifact.compatibility_digest: artifact
        for artifact in artifacts
        if isinstance(artifact, GraphArtifact)
    }
