"""The stores one PyTorch planning call reads and writes.

This module is intentionally policy-only.  It does not capture graphs, profile
tasks, construct Programs, or admit runtime memory.
"""

from __future__ import annotations

from dataclasses import dataclass

from shadowspill.planner.artifact_store import ArtifactStore
from shadowspill.planner.plan_store import PlanStore
from shadowspill.pytorch.capture.aot import ExportCapture, export_capture_digest
from shadowspill.pytorch.profiling import ProfileStore

from ..graph_pairs import GraphPairStore


@dataclass(frozen=True, slots=True)
class PlanningStores:
    """The stores this planning call may look in and write to."""

    store: ArtifactStore
    profiles: ProfileStore
    plans: PlanStore
    graph_pairs: GraphPairStore

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

def open_planning_stores(store: ArtifactStore) -> PlanningStores:
    """Open the stores one artifact-store policy implies."""

    return PlanningStores(
        store=store,
        profiles=ProfileStore(
            store.profile_measurements,
            compiled_manifest_root=store.compiled_manifests,
            read_enabled=store.read_enabled,
            write_enabled=store.write_enabled,
            overwrite=store.overwrite_plan,
            artifact_recorder=store.record,
        ),
        plans=PlanStore(
            store.pressurefit_selections,
            read_enabled=store.read_enabled,
            write_enabled=store.write_enabled,
            overwrite=store.overwrite_plan,
            artifact_recorder=store.record,
        ),
        graph_pairs=GraphPairStore(
            store.graphpairs,
            read_enabled=store.read_enabled,
            write_enabled=store.write_enabled,
            overwrite=store.overwrite_plan,
            artifact_recorder=store.record,
        ),
    )


__all__ = ["PlanningStores", "open_planning_stores"]
