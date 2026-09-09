"""The stores one PyTorch planning call reads and writes.

This module is intentionally policy-only.  It does not capture graphs, profile
tasks, construct Programs, or admit runtime memory.
"""

from __future__ import annotations

from dataclasses import dataclass

from shadowspill.planner.artifact_store import ArtifactStore
from shadowspill.planner.plan_store import PlanStore, open_plan_store
from shadowspill.pytorch.capture.aot import ExportCapture, export_capture_digest
from shadowspill.pytorch.profiling import ProfileStore

from ..graph_pairs import GraphPairStore
from ..optimizer.store import OptimizerCaptureStore


@dataclass(frozen=True, slots=True)
class PlanningStores:
    """The stores this planning call may look in and write to."""

    store: ArtifactStore
    profiles: ProfileStore
    plans: PlanStore
    graph_pairs: GraphPairStore
    optimizer_captures: OptimizerCaptureStore

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
    """Open the stores one artifact-store policy implies.

    Profiles, graph pairs and optimizer captures live in the build tree and
    follow ``build_store_mode``; plans live in the planning tree and follow
    ``plan_store_mode``. They were one switch until the two trees could be
    rooted apart, and one switch meant a run that kept its plans to itself
    also stopped contributing the builds it had paid for.
    """

    return PlanningStores(
        store=store,
        profiles=ProfileStore(
            store.profile_measurements,
            compiled_manifest_root=store.compiled_manifests,
            policy=store.build_policy,
            artifact_recorder=store.record,
        ),
        plans=open_plan_store(store),
        graph_pairs=GraphPairStore(
            store.graphpairs,
            policy=store.build_policy,
            artifact_recorder=store.record,
        ),
        optimizer_captures=OptimizerCaptureStore(
            store.optimizer_captures,
            policy=store.build_policy,
            artifact_recorder=store.record,
        ),
    )


__all__ = ["PlanningStores", "open_planning_stores"]
