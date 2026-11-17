"""Select assigned devices and bind prepared worker startup records."""

from __future__ import annotations

from uuid import uuid4

from shadowspill.diagnostics.progress import phase
from shadowspill.orchestration.launch.admission import (
    NodeAdmission,
    ResourcesUnavailable,
)
from shadowspill.server.records.requests import RunIdentity
from shadowspill.server.startup.weights import ResolvedWeights
from shadowspill.training.preparation.planning.records import PreparedPlans
from shadowspill.training.preparation.setup import ResolvedSetup
from shadowspill.training.session.checkpoint import ResolvedCheckpoint
from shadowspill.training.worker.checkpoint_records import RankRestore, RankWeights
from shadowspill.training.worker.protocol import WorkerScope
from shadowspill.training.worker.runtime import PreparedWorker
from shadowspill_client.protocol.requests import ArtifactRef


def select_devices(
    node: NodeAdmission, setup: ResolvedSetup, *, wait: bool
) -> tuple[str, ...]:
    """Choose compatible assigned devices before acquiring any phase owner."""
    ranks = setup.context.parallelism.world_size
    needed = int(setup.config.memory.device_gib * (1 << 30))
    if needed < 1:
        raise ValueError("worker device budget must contain at least one byte")
    compatible = [
        device for device in node.capacity.devices if device.capacity_bytes >= needed
    ]
    if len(compatible) < ranks:
        raise ValueError("requested worker count/budget cannot fit this assigned node")
    occupied = node.used.devices
    compatible.sort(key=lambda device: (device.id in occupied, device.id))
    devices = tuple(device.id for device in compatible[:ranks])
    if not wait and occupied.intersection(devices):
        raise ResourcesUnavailable("compatible assigned workers have live owners")
    return devices


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
