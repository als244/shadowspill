"""Select assigned devices and bind prepared worker startup records."""

from __future__ import annotations

from uuid import uuid4

from shadowspill.diagnostics.progress import phase
from shadowspill.orchestration.launch.admission import (
    NodeAdmission,
)
from shadowspill.orchestration.nodes.hardware import node_hardware
from shadowspill.orchestration.nodes.selection import select_hardware
from shadowspill.server.records.requests import RunIdentity
from shadowspill.server.startup.memory import automatic
from shadowspill.server.startup.weights import ResolvedWeights
from shadowspill.training.preparation.planning.records import PreparedPlans
from shadowspill.training.preparation.setup import ResolvedSetup
from shadowspill.training.session.checkpoint import ResolvedCheckpoint
from shadowspill.training.worker.checkpoint_records import RankRestore, RankWeights
from shadowspill.training.worker.protocol import WorkerScope
from shadowspill.training.worker.runtime import PreparedWorker
from shadowspill_client import Resources
from shadowspill_client.protocol.hardware import HardwareInventory
from shadowspill_client.protocol.requests import ArtifactRef


def select_devices(
    node: NodeAdmission,
    setup: ResolvedSetup,
    *,
    wait: bool,
    resources: Resources | None = None,
    pool: str = "default",
) -> tuple[str, ...]:
    """Choose compatible assigned devices before acquiring any phase owner."""
    if automatic(setup.config.memory):
        # Describing the run sizes them, and the run records the sizes.
        raise ValueError(
            "a run is created from settings with sized memory budgets; describe "
            "settings with an automatic (zero) budget first"
        )
    selection = select_hardware(
        HardwareInventory((node_hardware(node, pool=pool),)),
        ranks=setup.context.parallelism.world_size,
        device_bytes=int(setup.config.memory.device_gib * (1 << 30)),
        resources=resources,
        wait=wait,
    )
    return tuple(worker.device.id for worker in selection.workers)


def resolve_workers(
    family: PreparedPlans,
    plans: ArtifactRef,
    identity: RunIdentity,
    server_incarnation: str,
    *,
    ranks: int,
    weights: list[ArtifactRef],
    checkpoint: ResolvedCheckpoint | None,
    weights_source: ResolvedWeights | None,
    initial_weights: ResolvedWeights | None,
) -> dict[int, PreparedWorker]:
    """Bind saved state and rank identities before claiming worker resources."""
    with phase("Resolve worker initialization and storage"):
        workers = {
            rank: PreparedWorker.resolve(
                family,
                WorkerScope(
                    identity.run_id,
                    server_incarnation,
                    "worker-" + uuid4().hex,
                    rank,
                ),
                bundle=plans,
                rank_map={worker: worker for worker in range(ranks)},
                weights=weights,
                restore=None
                if checkpoint is None
                else RankRestore(
                    checkpoint.reference,
                    checkpoint.manifest.info,
                    checkpoint.manifest.ranks[rank],
                ),
                initial_weights=None
                if initial_weights is None or weights_source is None
                else RankWeights(
                    weights_source.reference,
                    initial_weights.reference,
                    initial_weights.manifest.info,
                    initial_weights.manifest.ranks[rank],
                ),
            )
            for rank in sorted(family.plans)
        }
    return workers
