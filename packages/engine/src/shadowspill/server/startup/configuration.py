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
from shadowspill.server.startup.model_sources import local_model_source
from shadowspill.server.storage import ServiceStorage
from shadowspill.training.definitions import load_registry
from shadowspill.training.preparation.setup import resolve_training
from shadowspill.training.worker.resources import WorkerHost
from shadowspill_client._records import integer, names, text
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.hardware import HardwareInventory
from shadowspill_client.protocol.preview import TrainingDescription, TrainingPreview
from shadowspill_client.protocol.runs import RunInfo

if TYPE_CHECKING:
    from .creation import LocalCreation


@dataclass(frozen=True)
class LocalCreationLimits:
    """Deployment-qualified envelopes; no estimates from tensor pools or idle RSS.

    The registry separately owns decoded-family/creation-thread host capacity.
    ``preparation_disk_bytes`` includes the owned phase directory and its caches;
    phase processes must be qualified within that envelope, like their host bound.
    File-count/size checks reject excess before publication or persistent startup.
    Shared external compiler caches remain the deployment's separate ownership.
    """

    preparation_host_bytes: int
    preparation_disk_bytes: int
    worker: WorkerHost
    transfer_bytes: int
    publication_bytes: int
    log_bytes: int
    max_preparation_files: int
    nsys: CollectorLimits = field(default_factory=CollectorLimits)

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


class LocalCreationFactory:
    """Configuration comes from deployment; per-run choices come from storage.

    This adapter serves one assigned node. It neither discovers remote capacity
    nor requests a scheduler allocation. Multi-node agents use the same lower
    level phase/worker interfaces and require their own allocation integration.
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
        self.packages = names(packages, "definition packages")
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
            raise ValueError("local allocation and remote agent have the same node ID")
        return HardwareInventory((local, *remote) if local.devices else remote)

    def startable(self) -> HardwareInventory:
        """The hardware a run can start on: this service's own node.

        ``inventory`` also lists registered nodes, which cannot yet run a
        training worker. Creation selects from this node alone, so a preview
        selects from the same hardware and never shows a placement that
        creation would not make.
        """
        return HardwareInventory((node_hardware(self.node, pool=self.pool),))

    def describe(self, request: TrainingDescription) -> TrainingPreview:
        """Use the same CPU resolver; the caller owns the control-memory envelope."""
        with checksum_cache(self.cache / "source-digests"):
            settings = request.config if request.config is not None else request.setup
            assert settings is not None
            setup = resolve_training(
                load_registry(self.packages),
                local_model_source(settings, self.model_root),
            )
        selection = select_hardware(
            self.startable(),
            ranks=setup.context.parallelism.world_size,
            device_bytes=int(setup.config.memory.device_gib * (1 << 30)),
            resources=request.resources,
            wait=True,
        )
        return replace(
            setup.preview(), resources=selection.resources, hardware=selection.workers
        )
