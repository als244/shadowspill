"""Deployment configuration for local training creation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

from shadowspill.checkpoint.file_digest import checksum_cache
from shadowspill.orchestration.capture.files import CollectorLimits
from shadowspill.orchestration.launch.admission import (
    NodeAdmission,
)
from shadowspill.orchestration.nodes.hardware import node_hardware
from shadowspill.orchestration.nodes.registry import NodeRegistry
from shadowspill.orchestration.nodes.selection import select_hardware
from shadowspill.server.startup.memory import (
    automatic,
    device_request_bytes,
    resolve_memory,
)
from shadowspill.server.startup.model_sources import local_model_source
from shadowspill.server.storage import ServiceStorage
from shadowspill.training.components import load_registry
from shadowspill.training.preparation.setup import ResolvedSetup, resolve_training
from shadowspill.training.worker.resources import WorkerHost
from shadowspill_client._records import integer, names, number, text
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.hardware import HardwareInventory, WorkerPlacement
from shadowspill_client.protocol.preview import TrainingDescription, TrainingPreview
from shadowspill_client.protocol.runs import RunInfo
from shadowspill_client.protocol.training import Origin

if TYPE_CHECKING:
    from .creation import LocalCreation


@dataclass(frozen=True)
class LocalCreationLimits:
    """Envelopes of one run's creation and workers, as `ServerLimits.creation`.

    A deployment qualifies them for the components it installs; none is
    estimated from pool sizes or idle memory use. A preparation that exceeds
    an envelope fails with the envelope named, and its directory is removed
    unpublished. Compiler caches shared outside the service are the
    deployment's own.

    Parameters
    ----------
    preparation_host_bytes : int
        Host memory, in bytes, of one preparation phase process, or of a
        process that converts saved state to new Plans.
    preparation_disk_bytes : int
        Disk space, in bytes, of one creation's working directory, its caches
        included; it also bounds the Plan bundle and exported measurements.
    worker : WorkerHost
        Host allowance of each training worker.
    transfer_bytes : int
        Host memory, in bytes, for streaming files between the service and
        its phase and worker processes.
    publication_bytes : int
        Host memory, in bytes, for publishing a live run's results, logs and
        manifests as artifacts.
    log_bytes : int
        Bytes kept of each output stream of each phase or worker process;
        the rest is dropped.
    max_preparation_files : int
        Entries a preparation's working directory may hold, profiling cache
        included. Workers whose programs differ keep one cache entry per
        worker and task.
    nsys : CollectorLimits, default CollectorLimits()
        Envelope of the Nsight Systems collector, for runs that capture.
    worker_startup_seconds : float, default 1800
        How long a started worker may take to report ready, in seconds. A
        worker binds its Plans and loads the run's model values in that
        time; one that exits is reported at once, so this bounds only one
        that hangs.

    Raises
    ------
    ValueError
        If an integer envelope is not an int of at least 1, or
        ``worker_startup_seconds`` is not a positive number.
    TypeError
        If ``worker`` is not a `WorkerHost` or ``nsys`` is not a
        `CollectorLimits`.
    """

    preparation_host_bytes: int
    preparation_disk_bytes: int
    worker: WorkerHost
    transfer_bytes: int
    publication_bytes: int
    log_bytes: int
    max_preparation_files: int
    nsys: CollectorLimits = field(default_factory=CollectorLimits)
    # A started worker binds its Plans and loads the run's model values before
    # it reports ready, so the wait grows with the checkpoint and its storage.
    # A worker that exits is seen at once; this bounds only one that hangs.
    worker_startup_seconds: float = 1800

    def __post_init__(self) -> None:
        for name in (
            "preparation_host_bytes",
            "preparation_disk_bytes",
            "transfer_bytes",
            "publication_bytes",
            "log_bytes",
            "max_preparation_files",
        ):
            integer(getattr(self, name), name, minimum=1)
        if not isinstance(self.worker, WorkerHost):
            raise TypeError("worker host envelope requires WorkerHost")
        if not isinstance(self.nsys, CollectorLimits):
            raise TypeError("Nsight deployment envelope requires CollectorLimits")
        if not number(self.worker_startup_seconds, "worker startup wait", minimum=0):
            raise ValueError("worker startup wait must be positive")


class LocalCreationFactory:
    """Configuration comes from deployment; per-run choices come from storage.

    This adapter serves one assigned node. It neither discovers remote capacity
    nor requests a scheduler allocation. The node daemons of a run that spans
    nodes use the same lower level phase/worker interfaces and require their
    own allocation integration.
    """

    def __init__(
        self,
        storage: ServiceStorage,
        node: NodeAdmission,
        *,
        packages: Sequence[str],
        root: Path,
        cache: Path,
        environment: Mapping[str, str],
        limits: LocalCreationLimits,
        disk: BufferBudget,
        base_host_bytes: int,
        model_root: Path | None = None,
        pool: str = "default",
        nodes: NodeRegistry | None = None,
    ) -> None:
        self.storage, self.node = storage, node
        self.pool = text(pool, "compute pool")
        self.nodes = nodes
        self.packages = names(packages, "component packages")
        self.root, self.cache = root, cache
        self.model_root = model_root
        self.environment = MappingProxyType(dict(environment))
        self.limits, self.disk = limits, disk
        # Permanent service arenas (including all registry job slots) cannot be
        # waited out. Include them when rejecting impossible combined demands.
        self.base_host_bytes = integer(base_host_bytes, "fixed service arena bytes")
        if base_host_bytes > node.capacity.host_bytes:
            raise ValueError("fixed service arenas exceed node capacity")

    def __call__(self, info: RunInfo) -> LocalCreation:
        from .creation import LocalCreation

        return LocalCreation(self, info)

    def inventory(self) -> HardwareInventory:
        local = node_hardware(self.node, pool=self.pool)
        if self.nodes is None:
            return HardwareInventory((local,))
        remote = self.nodes.inventory().nodes
        if any(node.id == local.id for node in remote) and local.devices:
            raise ValueError(
                "local allocation and remote node daemon have the same node ID"
            )
        return HardwareInventory((local, *remote) if local.devices else remote)

    def _reserved_host(
        self, setup: ResolvedSetup, workers: Sequence[WorkerPlacement]
    ) -> dict[str, int]:
        """Host memory the run holds on each node beside its spill pools."""
        from .creation import reserved_host_bytes

        ranks: dict[str, int] = {}
        for worker in workers:
            ranks[worker.node] = ranks.get(worker.node, 0) + 1
        return {
            node: reserved_host_bytes(
                self.limits,
                self.base_host_bytes,
                setup.config.execution.max_staging_bytes,
                count,
            )
            for node, count in ranks.items()
        }

    def startable(self) -> HardwareInventory:
        """The hardware a run can start on: this service's own node.

        ``inventory`` also lists registered nodes, which run no training
        worker. Creation selects from this node alone, so a preview
        selects from the same hardware and never shows a placement that
        creation would not make.
        """
        return HardwareInventory((node_hardware(self.node, pool=self.pool),))

    def describe(self, request: TrainingDescription) -> TrainingPreview:
        """Use the same CPU resolver; the caller owns the control-memory envelope.

        Automatic memory budgets are sized here, on the selected workers, and
        the preview's settings hold the sizes.
        """
        with checksum_cache(self.cache / "source-digests"):
            settings = request.config if request.config is not None else request.setup
            assert settings is not None
            setup = resolve_training(
                load_registry(self.packages),
                local_model_source(settings, self.model_root),
            )
        inventory = self.startable()
        memory = setup.config.memory
        selection = select_hardware(
            inventory,
            ranks=setup.context.parallelism.world_size,
            device_bytes=device_request_bytes(memory),
            resources=request.resources,
            wait=True,
        )
        preview = setup.preview()
        if automatic(memory):
            sized: dict[str, Origin] = {
                f"memory.{name}": "engine_default"
                for name in ("device_gib", "spill_gib")
                if getattr(memory, name) == 0
            }
            preview = replace(
                preview,
                config=replace(
                    preview.config,
                    memory=resolve_memory(
                        memory,
                        selection.workers,
                        inventory,
                        self._reserved_host(setup, selection.workers),
                    ),
                    origins={**preview.config.origins, **sized},
                ),
            )
        return replace(
            preview, resources=selection.resources, hardware=selection.workers
        )
