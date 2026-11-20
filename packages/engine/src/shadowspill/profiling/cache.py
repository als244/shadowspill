"""Portable per-task measurements, independent of candidate-wide memory costs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shadowspill.artifacts._values import digest, plain
from shadowspill.artifacts.profiles import TaskProfile
from shadowspill.store import (
    STORE_MODES,
    ArtifactStore,
    StoreMode,
    StorePolicy,
    atomic_json,
    digest_directory,
)
from shadowspill.store.artifacts import read_json

from .records import measured_task

_SCHEMA = "shadowspill.task-measurement/v1"


@dataclass(frozen=True)
class CachedTask:
    """One task measurement a `TaskProfileCache` holds, as its ``get`` returns.

    Attributes
    ----------
    profile : TaskProfile
        The raw measured profile.
    external_device_bytes : int
        Device memory the process used outside PyTorch's allocator when the
        task was measured, in bytes. It sets a floor under the fixed device
        memory of a Program that reuses the measurement.
    """

    profile: TaskProfile
    external_device_bytes: int


class TaskProfileCache:
    """A store of task measurements, reused across Programs and processes.

    Pass one to `measure_program`, the same instance across a candidate
    sweep. A measurement is reused only for the same finalized code, input
    geometry and lineage, setup, numerical settings, resources and rank
    layout, allocator configuration and timing bounds; one with more samples
    satisfies a smaller ``minimum_samples``. Only task timing and allocation
    evidence is reused: every `measure_program` call still loads its code and
    resources and measures its own fixed memory. Tasks that retain lazy
    allocations are never stored, and calls with explicit ``inputs`` bypass
    the cache.

    Parameters
    ----------
    root : str or pathlib.Path or ArtifactStore, optional
        The store, or its root directory. By default the user's store under
        ``~/.cache/shadowspill``.
    mode : str, default "contribute"
        ``"contribute"`` reads hits and writes new measurements; ``"reuse"``
        reads without writing; ``"require"`` makes `measure_program` raise
        `LookupError` on a miss; ``"refresh"`` ignores earlier entries on disk
        but reuses what this instance measured.

    Attributes
    ----------
    store : ArtifactStore
        The store the measurements live in.
    policy : StorePolicy
        What ``mode`` permits.

    Raises
    ------
    ValueError
        If ``mode`` is none of these.
    """

    def __init__(
        self,
        root: str | Path | ArtifactStore | None = None,
        *,
        mode: StoreMode = "contribute",
    ) -> None:
        if mode not in STORE_MODES:
            raise ValueError(f"invalid task profile cache mode: {mode!r}")
        self.store = (
            root if isinstance(root, ArtifactStore) else ArtifactStore.resolve(root)
        )
        self.policy = StorePolicy.for_mode(mode)
        self._records: dict[str, CachedTask] = {}

    def _path(self, key: str) -> Path:
        return digest_directory(self.store.task_profiles, key) / "measurement.json"

    def get(
        self,
        identity: Mapping[str, Any],
        manifest: Mapping[str, Any],
        *,
        minimum_samples: int,
    ) -> CachedTask | None:
        """Look up a task measurement.

        Parameters
        ----------
        identity : mapping
            The measurement's identity, as `measure_program` builds it.
        manifest : mapping
            The task form's saved manifest, which the stored evidence must
            fit.
        minimum_samples : int
            Timing samples the measurement must hold.

        Returns
        -------
        CachedTask or None
            The measurement; None on a miss, or when it holds fewer samples.

        Raises
        ------
        ValueError
            If the stored entry's identity differs, it is corrupt, or its
            evidence does not fit ``manifest``.
        """
        key = digest(identity)
        if key in self._records:
            result = self._records[key]
        elif self.policy.read_enabled and (path := self._path(key)).is_file():
            value = read_json(path)
            if value.get("schema") != _SCHEMA or value.get("identity") != plain(
                identity
            ):
                raise ValueError(f"task measurement cache identity differs: {path}")
            data = value.get("measurement")
            if not isinstance(data, dict) or digest(data) != value.get(
                "measurement_digest"
            ):
                raise ValueError(f"task measurement cache evidence is corrupt: {path}")
            if (
                not isinstance(data.get("profile"), dict)
                or type(data.get("external_device_bytes")) is not int
            ):
                raise ValueError(f"invalid task measurement cache fields: {path}")
            result = CachedTask(
                TaskProfile.from_dict(data["profile"]), data["external_device_bytes"]
            )
            self._validate(result, identity, manifest)
            self._records[key] = result
            self.store.record(
                category="profiling",
                kind="task_measurement",
                digest=key,
                path=path,
                access="read",
                schema=_SCHEMA,
            )
        else:
            return None
        if len(result.profile.metadata.get("samples_seconds", ())) < minimum_samples:
            return None
        return result

    def put(
        self,
        identity: Mapping[str, Any],
        manifest: Mapping[str, Any],
        profile: TaskProfile,
        external_device_bytes: int | None,
    ) -> bool:
        """Record a task measurement in this instance, and on disk if allowed.

        An entry on disk is replaced only in ``"refresh"`` mode or by a
        measurement with more timing samples.

        Parameters
        ----------
        identity : mapping
            The measurement's identity, as for `get`.
        manifest : mapping
            The task form's saved manifest.
        profile : TaskProfile
            The raw measured profile.
        external_device_bytes : int or None
            Device memory the process used outside PyTorch's allocator at
            measurement, in bytes; None when unknown.

        Returns
        -------
        bool
            Whether it was recorded. A measurement that retained lazy
            allocations, or whose ``external_device_bytes`` is unknown, is
            not.

        Raises
        ------
        ValueError
            If ``profile`` is not raw measured evidence fitting ``identity``
            and ``manifest``.
        """
        # Replaying a lazy allocation would be necessary to charge its lifetime.
        # Unknown external memory also cannot establish a safe cold-start floor.
        if (
            sum(profile.metadata.get("warmup_persistent_bytes", ()))
            or external_device_bytes is None
        ):
            return False
        result = CachedTask(profile, external_device_bytes)
        self._validate(result, identity, manifest)
        key = digest(identity)
        previous = self._records.get(key)
        more_samples = previous is not None and len(
            previous.profile.metadata.get("samples_seconds", ())
        ) < len(profile.metadata.get("samples_seconds", ()))
        self._records[key] = result
        if self.policy.write_enabled:
            path = self._path(key)
            if not path.is_file() or self.policy.overwrite or more_samples:
                measurement = {
                    "profile": profile.to_dict(),
                    "external_device_bytes": external_device_bytes,
                }
                atomic_json(
                    path,
                    {
                        "schema": _SCHEMA,
                        "identity": plain(identity),
                        "measurement": measurement,
                        "measurement_digest": digest(measurement),
                    },
                )
                self.store.record(
                    category="profiling",
                    kind="task_measurement",
                    digest=key,
                    path=path,
                    access="write",
                    schema=_SCHEMA,
                )
        return True

    @staticmethod
    def _validate(
        result: CachedTask, identity: Mapping[str, Any], manifest: Mapping[str, Any]
    ) -> None:
        profile = result.profile
        if (
            profile.compatibility_digest != identity["implementation"]
            or profile.metadata.get("memory_kind") != "measured"
            or profile.metadata.get("duration_kind") != "measured"
            or profile.duration_seconds is None
            or profile.workspace_bytes is None
            or profile.resource_phases
            or profile.overlap_costs
            or sum(profile.metadata.get("warmup_persistent_bytes", ()))
            or type(result.external_device_bytes) is not int
            or result.external_device_bytes < 0
        ):
            raise ValueError("task cache requires compatible raw measured evidence")
        measured_task(profile, manifest)
