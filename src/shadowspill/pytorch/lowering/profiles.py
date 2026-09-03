"""Shared physical-layout and task-profile resolution for PyTorch lowering."""

from __future__ import annotations

from collections.abc import Mapping

from shadowspill.errors import CaptureError
from shadowspill.ir import TaskProfile
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.capture.storage import TaskStorageContract
from shadowspill.pytorch.compilation.inductor import ExecutableRootAllocation
from shadowspill.pytorch.compilation.layout import (
    CompiledTaskLayout,
    reconcile_compiled_task_layout,
    replacement_transition_bytes,
)
from shadowspill.pytorch.optimizer import OptimizerTaskArtifact
from shadowspill.pytorch.profiling import TaskMeasurement

ProfileMeasurementKey = str | tuple[str, str | None]
ProfiledArtifact = GraphArtifact | OptimizerTaskArtifact


class CompiledLayoutIndex:
    """Deduplicate immutable compiled layouts within one lowering call."""

    def __init__(self) -> None:
        self._layouts: dict[tuple[str, str], CompiledTaskLayout] = {}

    def resolve(
        self,
        artifact: GraphArtifact,
        contract: TaskStorageContract,
        measurement: TaskMeasurement,
        root_allocations: tuple[ExecutableRootAllocation, ...] | None = None,
    ) -> CompiledTaskLayout:
        candidate = reconcile_compiled_task_layout(
            contract,
            measurement,
            root_allocations=root_allocations,
        )
        key = artifact.compatibility_digest, candidate.compatibility_digest
        existing = self._layouts.get(key)
        if existing is None:
            self._layouts[key] = candidate
            return candidate
        if existing.contract_digest != contract.compatibility_digest:
            raise CaptureError("one physical profile resolved to several contracts")
        return existing


class TaskProfileCatalog:
    """Resolve one shared set of semantic contracts and physical profiles."""

    def __init__(
        self,
        measurements: Mapping[ProfileMeasurementKey, TaskMeasurement],
        *,
        storage_contracts: Mapping[str, TaskStorageContract] | None = None,
        root_allocations: Mapping[str, tuple[ExecutableRootAllocation, ...]]
        | None = None,
        compatibility_digests: Mapping[tuple[str, str | None], str] | None = None,
        metadata_enabled: bool = False,
        layout_cache: CompiledLayoutIndex | None = None,
    ) -> None:
        self._measurements = measurements
        self._storage_contracts = storage_contracts
        self._root_allocations = root_allocations
        self._compatibility_digests = compatibility_digests
        self._metadata_enabled = metadata_enabled
        self._layout_cache = layout_cache or CompiledLayoutIndex()
        self._profile_by_key: dict[str, str] = {}
        self._profiles: list[TaskProfile] = []

    @property
    def profiles(self) -> tuple[TaskProfile, ...]:
        return tuple(self._profiles)

    def contract(self, artifact: GraphArtifact) -> TaskStorageContract:
        if self._storage_contracts is None:
            return artifact.storage_contract
        try:
            return self._storage_contracts[artifact.compatibility_digest]
        except KeyError as exc:
            raise CaptureError(
                "compiled storage contract is missing for artifact "
                f"{artifact.compatibility_digest}"
            ) from exc

    def measurement(
        self,
        artifact: ProfiledArtifact,
        metadata_digest: str | None = None,
    ) -> TaskMeasurement:
        measurement = self._measurements.get(
            (
                artifact.compatibility_digest,
                metadata_digest,
            )
        )
        if measurement is None and not self._metadata_enabled:
            measurement = self._measurements.get(artifact.compatibility_digest)
        if measurement is None:
            raise CaptureError(
                "profile scatter is missing "
                f"artifact={artifact.compatibility_digest}, "
                f"profiling_metadata={metadata_digest}"
            )
        return measurement

    def profile_id(
        self,
        artifact: ProfiledArtifact,
        extra_workspace: int = 0,
        *,
        metadata_digest: str | None = None,
    ) -> str:
        measurement = self.measurement(artifact, metadata_digest)
        compatibility_digest = self._compatibility_digest(
            artifact,
            metadata_digest,
        )
        key = (
            f"{compatibility_digest}:{measurement.runtime_ns}:"
            f"{measurement.workspace_charged_bytes}:{extra_workspace}"
        )
        existing = self._profile_by_key.get(key)
        if existing is not None:
            return existing
        profile_id = f"profile_{len(self._profiles):06d}"
        self._profile_by_key[key] = profile_id
        self._profiles.append(
            TaskProfile(
                profile_id,
                measurement.runtime_ns,
                measurement.workspace_charged_bytes + extra_workspace,
                compatibility_digest,
            )
        )
        return profile_id

    def layout(
        self,
        artifact: GraphArtifact,
        metadata_digest: str | None = None,
    ) -> CompiledTaskLayout:
        return self._layout_cache.resolve(
            artifact,
            self.contract(artifact),
            self.measurement(artifact, metadata_digest),
            self._root_allocations_for(artifact),
        )

    def mutation_transition_bytes(
        self,
        artifact: GraphArtifact,
        metadata_digest: str | None = None,
    ) -> int:
        return replacement_transition_bytes(
            self.contract(artifact),
            self.layout(artifact, metadata_digest),
        )

    def additional_workspace_for_outputs(
        self,
        artifact: GraphArtifact,
        leaf_indices: tuple[int, ...],
        metadata_digest: str | None = None,
    ) -> int:
        """Return the measured peak increase from transient task outputs."""

        return self.layout(
            artifact,
            metadata_digest,
        ).additional_workspace_for_outputs(leaf_indices)

    def replacement_output_leaves(
        self,
        artifact: GraphArtifact,
    ) -> tuple[int, ...]:
        """Return fresh leaves that replace an existing input generation."""

        return tuple(
            mutation.replacement_output_leaf
            for mutation in self.contract(artifact).mutations
            if mutation.replacement_output_leaf is not None
        )

    def _compatibility_digest(
        self,
        artifact: ProfiledArtifact,
        metadata_digest: str | None,
    ) -> str:
        if self._compatibility_digests is None:
            return artifact.compatibility_digest
        try:
            return self._compatibility_digests[
                (
                    artifact.compatibility_digest,
                    metadata_digest,
                )
            ]
        except KeyError as exc:
            raise CaptureError(
                "profile identity is missing "
                f"artifact={artifact.compatibility_digest}, "
                f"profiling_metadata={metadata_digest}"
            ) from exc

    def _root_allocations_for(
        self,
        artifact: GraphArtifact,
    ) -> tuple[ExecutableRootAllocation, ...] | None:
        if self._root_allocations is None:
            return None
        try:
            return self._root_allocations[artifact.compatibility_digest]
        except KeyError as exc:
            raise CaptureError(
                "compiled root allocations are missing for artifact "
                f"{artifact.compatibility_digest}"
            ) from exc


__all__ = [
    "CompiledLayoutIndex",
    "ProfileMeasurementKey",
    "TaskProfileCatalog",
]
