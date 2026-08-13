"""Cache-backed artifacts used by the PyTorch planning pipeline.

Planning sessions consume this facade instead of constructing individual
cache implementations or interpreting cache policy flags.  Semantic planning
code therefore sees one store with explicit operations at artifact boundaries.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from shadowspill.ir import Program, ResidencySpec
from shadowspill.planner import PressureFitOptions
from shadowspill.planner._cache import CachedPressureFitResult, PressureFitCache
from shadowspill.simulator import SimulationConfig

from ._planning_cache import PlanningCache
from .aot import ExportCapture, export_capture_digest
from .partition import _TrainingGraphPairCache
from .profiling import ProfileCache


@dataclass(frozen=True, slots=True)
class PlanningArtifacts:
    """All persistent artifact services for one planning call."""

    cache: PlanningCache
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
        self.cache.archive_export(
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

    def select(
        self,
        program: Program,
        *,
        initial_residency: tuple[ResidencySpec, ...],
        final_residency: tuple[ResidencySpec, ...],
        config: SimulationConfig,
        options: PressureFitOptions | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> CachedPressureFitResult:
        """Archive ``program`` and resolve its exact PressureFit selection."""

        self.cache.archive_program(program)
        return self.pressurefit.resolve(
            program,
            initial_residency=initial_residency,
            final_residency=final_residency,
            config=config,
            options=options,
            progress=progress,
        )


def planning_artifacts(cache: PlanningCache) -> PlanningArtifacts:
    """Create one policy-complete artifact facade for a planning call.

    This is the only place that translates :class:`PlanningCache` policy into
    graph-pair, profiling, and PressureFit cache implementations.
    """

    return PlanningArtifacts(
        cache=cache,
        profiles=ProfileCache(
            cache.profile_measurements,
            compiled_manifest_root=cache.compiled_manifests,
            read_enabled=cache.read_enabled,
            write_enabled=cache.write_enabled,
            overwrite=cache.overwrite_plan,
            artifact_recorder=cache.record,
        ),
        pressurefit=PressureFitCache(
            cache.pressurefit_selections,
            read_enabled=cache.read_enabled,
            write_enabled=cache.write_enabled,
            overwrite=cache.overwrite_plan,
            artifact_recorder=cache.record,
        ),
        graph_pairs=_TrainingGraphPairCache(
            cache.graphpairs,
            read_enabled=cache.read_enabled,
            write_enabled=cache.write_enabled,
            overwrite=cache.overwrite_plan,
            artifact_recorder=cache.record,
        ),
    )


__all__ = ["PlanningArtifacts", "planning_artifacts"]
