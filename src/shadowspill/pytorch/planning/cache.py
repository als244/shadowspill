"""Artifact lookup and persistence for PyTorch planning.

This module is intentionally policy-only.  It does not capture graphs, profile
tasks, construct Programs, or admit runtime memory.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from shadowspill.ir import Program, ResidencySpec
from shadowspill.planner import PressureFitOptions
from shadowspill.planner._cache import CachedPressureFitResult, PressureFitCache
from shadowspill.simulator import SimulationConfig

from .._planning_cache import PlanningCache
from ..aot import ExportCapture, export_capture_digest
from ..partition import _TrainingGraphPairCache
from ..profiling import ProfileCache


@dataclass(frozen=True, slots=True)
class PlanningArtifactCache:
    """Typed lookup/write boundary for one planning call's artifacts."""

    store: PlanningCache
    profiles: ProfileCache
    pressurefit: PressureFitCache
    graph_pairs: _TrainingGraphPairCache

    def archive_export(
        self,
        capture: ExportCapture,
        *,
        mode: str,
        position: int,
    ) -> str:
        """Archive one freshly captured Export program and return its digest."""

        digest = export_capture_digest(capture)
        signature = capture.exported_program.graph_signature
        self.store.archive_export(
            capture.exported_program,
            digest=digest,
            metadata={
                "mode": mode,
                "position": position,
                "input_specs": [
                    {
                        "kind": item.kind.name,
                        "target": item.target,
                        "argument": getattr(item.arg, "name", None),
                    }
                    for item in signature.input_specs
                ],
                "output_specs": [
                    {
                        "kind": item.kind.name,
                        "target": item.target,
                        "argument": getattr(item.arg, "name", None),
                    }
                    for item in signature.output_specs
                ],
            },
        )
        return digest

    def resolve_pressurefit(
        self,
        program: Program,
        *,
        initial_residency: tuple[ResidencySpec, ...],
        final_residency: tuple[ResidencySpec, ...],
        config: SimulationConfig,
        options: PressureFitOptions | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> CachedPressureFitResult:
        """Return a validated selection, running PressureFit only on a miss."""

        self.store.archive_program(program)
        return self.pressurefit.resolve(
            program,
            initial_residency=initial_residency,
            final_residency=final_residency,
            config=config,
            options=options,
            progress=progress,
        )


def open_artifact_cache(store: PlanningCache) -> PlanningArtifactCache:
    """Translate one public cache policy into typed artifact lookups."""

    return PlanningArtifactCache(
        store=store,
        profiles=ProfileCache(
            store.profile_measurements,
            compiled_manifest_root=store.compiled_manifests,
            read_enabled=store.read_enabled,
            write_enabled=store.write_enabled,
            overwrite=store.overwrite_plan,
            artifact_recorder=store.record,
        ),
        pressurefit=PressureFitCache(
            store.pressurefit_selections,
            read_enabled=store.read_enabled,
            write_enabled=store.write_enabled,
            overwrite=store.overwrite_plan,
            artifact_recorder=store.record,
        ),
        graph_pairs=_TrainingGraphPairCache(
            store.graphpairs,
            read_enabled=store.read_enabled,
            write_enabled=store.write_enabled,
            overwrite=store.overwrite_plan,
            artifact_recorder=store.record,
        ),
    )


__all__ = ["PlanningArtifactCache", "open_artifact_cache"]
