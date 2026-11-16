"""Configured local creation using the existing preparation and live-run engines."""

from __future__ import annotations

import hashlib
import io
import json
import os
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Lock
from types import MappingProxyType
from typing import cast
from uuid import uuid4

from shadowspill.orchestration.capture.files import CollectorLimits
from shadowspill.orchestration.capture.nsys import NsysCapture
from shadowspill.orchestration.launch.admission import (
    NodeAdmission,
    NodeLease,
    ResourceRequest,
    ResourcesUnavailable,
)
from shadowspill.orchestration.launch.host_arena import HostArena, HostPartition
from shadowspill.orchestration.launch.log_files import LogFiles
from shadowspill.orchestration.launch.phases import NodePhases
from shadowspill.orchestration.launch.processes import worker_command
from shadowspill.training.definitions import load_registry
from shadowspill.training.preparation.bundle import TrainingBundle
from shadowspill.training.preparation.job import BUNDLE_NAME, measure_job, prepare_job
from shadowspill.training.preparation.measurement_export import export_measurements
from shadowspill.training.preparation.setup import resolve_training
from shadowspill.training.session.checkpoint import ResolvedCheckpoint
from shadowspill.training.worker.checkpoint_records import RankRestore
from shadowspill.training.worker.initialization import (
    prepare_frozen_initialization,
    prepare_initialization,
    restore_staging,
)
from shadowspill.training.worker.node import WorkerNode
from shadowspill.training.worker.protocol import WorkerScope
from shadowspill.training.worker.resources import WorkerHost
from shadowspill.training.worker.runtime import PreparedWorker
from shadowspill_client._records import integer, names
from shadowspill_client.buffers import BufferBudget, BufferLease
from shadowspill_client.protocol.preparation import PREPARATION_MEDIA_TYPE
from shadowspill_client.protocol.preview import TrainingPreview
from shadowspill_client.protocol.requests import ArtifactRef
from shadowspill_client.protocol.runs import CreationStage, RunInfo
from shadowspill_client.protocol.setup import TrainingSetup
from shadowspill_client.protocol.training import ResolvedTraining

from .artifacts import ArtifactStore
from .checkpoint_conversion import prepare_checkpoint
from .checkpoint_restore import CheckpointResolver
from .creation_records import load_creation
from .live import LiveRun
from .plan_summaries import publish_summary
from .preparation_reuse import stage_preparation_reuse
from .records.requests import RunIdentity
from .storage import ServiceStorage


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
    ) -> None:
        self.storage, self.node = storage, node
        self.packages = names(packages, "definition packages")
        self.root, self.cache = root, cache
        self.environment = MappingProxyType(dict(environment))
        self.limits, self.disk = limits, disk
        # Permanent service arenas (including all registry job slots) cannot be
        # waited out. Include them when rejecting impossible combined demands.
        self.base_host_bytes = integer(base_host_bytes, "fixed service arena bytes")
        if base_host_bytes > node.capacity.host_bytes:
            raise ValueError("fixed service arenas exceed node capacity")

    def __call__(self, info: RunInfo) -> LocalCreation:
        return LocalCreation(self, info)

    def describe(self, settings: TrainingSetup | ResolvedTraining) -> TrainingPreview:
        """Use the same CPU resolver; the caller owns the control-memory envelope."""
        return resolve_training(load_registry(self.packages), settings).preview()


class LocalCreation:
    """One resource-free factory result; RunJobs owns its thread and host envelope."""

    def __init__(self, factory: LocalCreationFactory, info: RunInfo) -> None:
        if (
            info.server_incarnation != factory.storage.incarnation
            or info.state != "preparing"
        ):
            raise ValueError("creation requires a fresh local server registration")
        self._factory, self.info = factory, info
        self.identity = RunIdentity(info.owner_id, info.session_id, info.run_id)
        self.run: LiveRun | None = None
        self._cancelled, self._starting, self._retiring = Event(), Lock(), Lock()
        self._started = self.closed = False
        self._pin: AbstractContextManager[ArtifactStore] | None = None
        self._artifacts: ArtifactStore | None = None
        self._arena: HostArena | None = None
        self._partitions: list[HostPartition] = []
        self._scratch: BufferLease | None = None
        self._directory: TemporaryDirectory[str] | None = None
        self._phases: NodePhases | None = None
        self._node: WorkerNode | None = None
        self._conversion_lease: NodeLease | None = None
        self._conversion_logs: LogFiles | None = None

    def _active(self) -> None:
        if self._cancelled.is_set() or self.closed:
            raise RuntimeError("creation ownership has stopped")

    def _progress(self, phase: str) -> None:
        self._active()
        if phase not in ("build", "profile", "communicate", "plan"):
            raise ValueError("creation cannot launch an execution recipe phase")
        self._factory.storage.catalog.progress(
            self.identity, cast(CreationStage, phase)
        )

    def _partition(self, size: int, count: int) -> HostPartition:
        assert self._arena is not None
        partition = self._arena.partition(size, count)
        self._partitions.append(partition)
        return partition

    def _admit_host(self, size: int, count: int, *, wait: bool) -> None:
        factory = self._factory
        if factory.base_host_bytes + size > factory.node.capacity.host_bytes:
            raise ValueError(
                "combined service and creation host demand cannot fit this node"
            )
        self._active()
        self._arena = HostArena(
            factory.node, size, count, wait=wait, cancelled=self._cancelled
        )

    def _release_host(self) -> None:
        for partition in self._partitions:
            partition.close()
        if self._arena is not None:
            self._arena.close()
        self._partitions.clear()
        self._arena = None

    def start(self) -> None:
        if not self._starting.acquire(blocking=False):
            raise RuntimeError("creation is already in progress")
        try:
            self._active()
            if self._started:
                raise RuntimeError("creation only starts once")
            self._started = True
            factory, limits = self._factory, self._factory.limits
            self._pin = factory.storage.artifacts(self.identity.owner_id)
            self._artifacts = self._pin.__enter__()
            creation = load_creation(self._artifacts, self.info)
            execution = creation.config.execution
            capture = (
                NsysCapture.from_settings(execution.nsys)
                if creation.kind == "training"
                else None
            )
            if capture is not None:
                # Resolve the installed executable before phases or device admission.
                replace(capture, executable=limits.nsys.executable).command(
                    (), output="unused"
                )
            registry = load_registry(factory.packages)
            setup = resolve_training(registry, config=creation.config)
            checkpoint = (
                None
                if creation.checkpoint is None
                else CheckpointResolver(
                    self._artifacts,
                    session_id=self.identity.session_id,
                    world_size=None,
                    configuration=setup.config,
                ).resolve(creation.checkpoint)
            )
            # A complete checkpoint is the initializer. Historical fresh-weight
            # provenance in saved Plans never requires opening that old source.
            if checkpoint is None and creation.kind == "training":
                prepare_initialization(setup)
            elif (
                checkpoint is not None
                and checkpoint.manifest.info.frozen_base is not None
            ):
                for rank in range(setup.context.parallelism.world_size):
                    initial_base = prepare_frozen_initialization(
                        setup, checkpoint.manifest.info.frozen_base, rank
                    )
                    restore_staging(initial_base, execution.max_staging_bytes)
            weights = []
            initial = creation.config.model.initialize
            if (
                checkpoint is None
                and creation.kind == "training"
                and initial is not None
                and initial.kind == "artifact"
            ):
                assert initial.reference is not None
                ref = self._artifacts.get(initial.reference)
                if ref is None:
                    raise ValueError(
                        "initial weights are not in this owner's artifact storage"
                    )
                weights.append(ref)
            if checkpoint is not None:
                base = checkpoint.manifest.info.frozen_base
                if base is not None and base.artifact is not None:
                    weights.append(base.artifact)
            ranks = setup.context.parallelism.world_size
            needed = int(setup.config.memory.device_gib * (1 << 30))
            if needed < 1:
                raise ValueError("worker device budget must contain at least one byte")
            compatible = [
                device
                for device in factory.node.capacity.devices
                if device.capacity_bytes >= needed
            ]
            if len(compatible) < ranks:
                raise ValueError(
                    "requested worker count/budget cannot fit this assigned node"
                )
            occupied = factory.node.used.devices
            compatible.sort(key=lambda device: (device.id in occupied, device.id))
            devices = tuple(device.id for device in compatible[:ranks])
            if not creation.wait_for_resources and occupied.intersection(devices):
                raise ResourcesUnavailable(
                    "compatible assigned workers have live owners"
                )
            command_bytes = 16 * (1 << 20) * ranks
            arena_bytes = (
                limits.transfer_bytes
                + limits.publication_bytes
                + command_bytes
                + execution.max_staging_bytes
            )
            self._active()
            self._scratch = factory.disk.reserve(
                limits.preparation_disk_bytes, timeout=0
            )
            self._directory = TemporaryDirectory(prefix="creation-", dir=factory.root)
            root = Path(self._directory.name)
            job_root = root / "preparation"
            if creation.plans is None:
                self._admit_host(
                    limits.transfer_bytes + limits.preparation_host_bytes,
                    2,
                    wait=creation.wait_for_resources,
                )
                transfers = self._partition(
                    limits.transfer_bytes, 16 * ranks + 32
                ).budget
                phase_host = self._partition(limits.preparation_host_bytes, 1)
                self._phases = NodePhases(
                    factory.node,
                    devices=devices,
                    host_bytes=limits.preparation_host_bytes,
                    root=root,
                    disk=factory.disk,
                    memory=transfers,
                    log_limit_bytes=limits.log_bytes,
                    wait_for_resources=creation.wait_for_resources,
                    admitted=self._progress,
                    host=phase_host,
                )
                context = {
                    "server_incarnation": factory.storage.incarnation,
                    "devices": list(devices),
                }
                stage_preparation_reuse(
                    self._artifacts,
                    factory.storage.catalog,
                    creation.config,
                    job_root,
                    packages=factory.packages,
                    context=context,
                    max_disk_bytes=limits.preparation_disk_bytes,
                    max_files=limits.max_preparation_files,
                )
                if creation.kind == "preparation":
                    measure_job(
                        job_root,
                        packages=factory.packages,
                        config=creation.config,
                        environment=factory.environment,
                        launch=self._phases,
                    )
                    self._active()
                    exported = export_measurements(
                        job_root,
                        root / "measurements.zip",
                        max_disk_bytes=limits.preparation_disk_bytes,
                        max_files=limits.max_preparation_files,
                        context=context,
                    )
                    reference = self._publish_file(
                        exported, "measurements", media_type=PREPARATION_MEDIA_TYPE
                    )
                    factory.storage.catalog.prepared(self.identity, reference)
                    return
                bundle = prepare_job(
                    job_root,
                    packages=factory.packages,
                    config=creation.config,
                    environment=factory.environment,
                    launch=self._phases,
                )
                self._check_files()
                plans = self._publish_file(job_root / BUNDLE_NAME, "bundle")
                self._archive_preparation()
                self._phases.close()
                self._phases = None
                # Decoded results belong to the registry's host envelope. All
                # temporary process/log owners retire before execution admission.
                self._release_host()
            else:
                plans = creation.plans
                if self._artifacts.get(plans.id) != plans:
                    raise ValueError(
                        "saved Plans are not in this owner's artifact storage"
                    )
                # Use the existing bounded immutable artifact stream. The archive
                # loader checks decompressed metadata/payload limits before decoding.
                if plans.size_bytes > limits.preparation_disk_bytes:
                    raise ValueError("saved Plans exceed the preparation disk envelope")
                path = root / BUNDLE_NAME
                with self._artifacts.open(plans) as source, path.open("wb") as target:
                    digest = hashlib.sha256()
                    while block := source.read(64 << 10):
                        target.write(block)
                        digest.update(block)
                if digest.hexdigest() != plans.sha256:
                    raise ValueError("saved Plans failed their content checksum")
                bundle = TrainingBundle.load(path)
                if bundle.config != creation.config:
                    raise ValueError(
                        "saved Plans differ from the exact resolved configuration"
                    )
            self._active()
            family = bundle.recheck(registry)
            publish_summary(self._artifacts, plans, family)
            if checkpoint is not None and checkpoint.manifest.info.plans != plans:
                checkpoint = self._convert_checkpoint(
                    checkpoint,
                    plans,
                    setup.config,
                    ranks,
                    root,
                    wait=creation.wait_for_resources,
                )
            workers = {
                rank: PreparedWorker.resolve(
                    family,
                    WorkerScope(
                        self.identity.run_id,
                        factory.storage.incarnation,
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
                )
                for rank in sorted(family.plans)
            }
            worker_bytes = sum(
                limits.worker.peak_bytes(worker.startup) for worker in workers.values()
            )
            if capture is not None:
                worker_bytes += ranks * limits.nsys.host_bytes
            self._admit_host(
                arena_bytes + worker_bytes, 5, wait=creation.wait_for_resources
            )
            transfers = self._partition(limits.transfer_bytes, 16 * ranks + 32).budget
            publication = self._partition(limits.publication_bytes, 3).budget
            metadata = self._partition(command_bytes, 2).budget
            uploads = self._partition(
                execution.max_staging_bytes, execution.max_pending_requests
            ).budget
            worker_host = self._partition(worker_bytes, 1)
            self._node = WorkerNode(
                factory.node,
                workers,
                dict(zip(workers, devices, strict=True)),
                dict.fromkeys(workers, limits.worker),
                packages=factory.packages,
                weights=weights,
                root=root,
                cache=factory.cache,
                disk=factory.disk,
                memory=transfers,
                # This adapter owns all ranks on one host. A fresh, run-owned
                # file rendezvous avoids sharing the allocation's launcher port
                # with preparation phases or another live training group.
                environment={
                    **factory.environment,
                    "SHADOWSPILL_RENDEZVOUS": (root / "rendezvous").as_uri(),
                },
                log_limit_bytes=limits.log_bytes,
                host_memory=worker_host,
                capture_limits=None if capture is None else limits.nsys,
            )
            self.run = LiveRun(
                factory.storage,
                self.identity,
                family,
                [self._node],
                plans=plans,
                history_id="history-" + uuid4().hex,
                checkpoint=checkpoint,
                parent_checkpoint=creation.checkpoint,
                root=root,
                metadata=metadata,
                uploads=uploads,
                publication=publication,
                disk=factory.disk,
            )
            self._active()
            self.run.start(wait_for_resources=creation.wait_for_resources)
        finally:
            self._starting.release()

    def _key(self, name: str) -> str:
        return hashlib.sha256(
            "\0".join((self.identity.session_id, self.identity.run_id, name)).encode()
        ).hexdigest()

    def _convert_checkpoint(
        self,
        checkpoint: ResolvedCheckpoint,
        plans: ArtifactRef,
        config: ResolvedTraining,
        world_size: int,
        root: Path,
        *,
        wait: bool,
    ) -> ResolvedCheckpoint:
        factory, limits = self._factory, self._factory.limits
        assert self._artifacts is not None
        if limits.preparation_host_bytes < 1 << 20:
            raise ValueError("checkpoint conversion host envelope is too small")
        self._admit_host(
            limits.preparation_host_bytes + limits.transfer_bytes, 2, wait=wait
        )
        try:
            host = self._partition(limits.preparation_host_bytes, 1)
            transfers = self._partition(limits.transfer_bytes, 4).budget
            self._conversion_lease = factory.node.reserve(
                ResourceRequest({}, limits.preparation_host_bytes),
                host=host,
                cancelled=self._cancelled,
            )
            self._conversion_logs = LogFiles(
                self._conversion_lease,
                None,
                root=root,
                disk=factory.disk,
                memory=transfers,
                limit_bytes=limits.log_bytes,
            )
            factory.storage.catalog.progress(self.identity, "converting")
            child = self._conversion_lease.launch_host(
                lambda fd: worker_command(
                    "shadowspill.training.worker.convert_checkpoint", (fd,)
                ),
                environment=factory.environment,
                directory=self._conversion_logs.directory,
                cwd=root,
                log_limit_bytes=limits.log_bytes,
            )
            return prepare_checkpoint(
                self._artifacts,
                checkpoint,
                child,
                plans=plans,
                configuration=self.info.configuration,
                config=config,
                world_size=world_size,
                host_bytes=limits.preparation_host_bytes,
                root=root,
                memory=transfers,
                disk=factory.disk,
                max_disk_bytes=limits.preparation_disk_bytes,
                max_files=limits.max_preparation_files,
            )
        finally:
            self._retire_conversion()
            self._release_host()

    def _retire_conversion(self) -> None:
        if self._conversion_lease is not None:
            self._conversion_lease.close()
        if self._conversion_logs is not None:
            assert self._artifacts is not None
            references = self._conversion_logs.references()
            for reference in references.values():
                self._artifacts.put(reference, self._conversion_logs.open)
            self._publish_metadata(
                "conversion",
                {
                    **self.identity.to_dict(),
                    "phase": "converting",
                    "streams": {
                        name: reference.to_dict()
                        for name, reference in references.items()
                    },
                },
            )
            self._conversion_logs.close()
            self._conversion_logs = None
        self._conversion_lease = None

    def _publish_file(
        self, path: Path, name: str, *, media_type: str = "application/octet-stream"
    ) -> ArtifactRef:
        assert self._artifacts is not None
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        reference = ArtifactRef(
            "prepared." + self._key(name), digest, path.stat().st_size, media_type
        )
        self._artifacts.put(reference, lambda _: path.open("rb"))
        return reference

    def _check_files(self) -> tuple[Path, ...]:
        if self._directory is None:
            return ()
        root = Path(self._directory.name) / "preparation"
        files, total, count = [], 0, 0
        if not root.exists():
            return ()
        for directory, children, filenames in os.walk(root, followlinks=False):
            for name in (*children, *filenames):
                path = Path(directory) / name
                if path.is_symlink():
                    raise ValueError(
                        "preparation artifacts cannot contain symbolic links"
                    )
                count += 1
                if count > self._factory.limits.max_preparation_files:
                    raise ValueError(
                        "preparation exceeds its configured file-count envelope"
                    )
                total += 4096 + (path.stat().st_size if path.is_file() else 0)
                if total > self._factory.limits.preparation_disk_bytes:
                    raise ValueError("preparation exceeds its configured disk envelope")
                if path.is_file() and path.suffix == ".ssp":
                    files.append(path)
        return tuple(sorted(files))

    def interrupt(self) -> None:
        self._cancelled.set()
        errors = []
        for stop in (
            None if self._conversion_lease is None else self._conversion_lease.close,
            None if self._phases is None else self._phases.stop,
            self.run.interrupt
            if self.run is not None
            else None
            if self._node is None
            else self._node.stop,
        ):
            if stop is not None:
                try:
                    stop()
                except BaseException as error:
                    errors.append(error)
        if errors:
            raise BaseExceptionGroup(
                "creation participants still require shutdown", errors
            )

    def _publish_metadata(self, name: str, value: dict[str, object]) -> None:
        assert self._artifacts is not None
        body = json.dumps(value, separators=(",", ":")).encode()
        if len(body) > 16 << 10:
            raise ValueError(
                "preparation manifest exceeds its per-record metadata bound"
            )
        reference = ArtifactRef(
            "preparation." + self._key(name),
            hashlib.sha256(body).hexdigest(),
            len(body),
            "application/json",
        )
        self._artifacts.put(reference, lambda _: io.BytesIO(body))

    def _archive_preparation(self) -> None:
        if self._artifacts is not None:
            for path in self._check_files():
                assert self._directory is not None
                relative = str(
                    path.relative_to(Path(self._directory.name) / "preparation")
                )
                ref = self._publish_file(path, relative)
                self._publish_metadata(
                    relative,
                    {
                        **self.identity.to_dict(),
                        "path": relative,
                        "artifact": ref.to_dict(),
                    },
                )
            if self._phases is not None:
                for phase, logs in self._phases.logs.items():
                    if logs.closed or not logs.directory.exists():
                        continue
                    references = {}
                    for stream, ref in logs.references().items():
                        self._artifacts.put(ref, logs.open)
                        references[stream] = ref.to_dict()
                    self._publish_metadata(
                        "phase:" + phase,
                        {
                            **self.identity.to_dict(),
                            "phase": phase,
                            "streams": references,
                        },
                    )

    def close(self) -> None:
        with self._retiring:
            if self.closed:
                return
            if self._starting.locked():
                raise RuntimeError("join creation before retiring its artifacts")
            if self.run is not None:
                self.run.close()
            elif self._node is not None:
                self._node.close()
            if self._phases is not None:
                self._phases.stop()
            self._retire_conversion()
            self._archive_preparation()
            if self._phases is not None:
                self._phases.close()
            if self._directory is not None:
                self._directory.cleanup()
            self._phases = self._node = None
            self._release_host()
            if self._scratch is not None:
                self._scratch.close()
            if self._artifacts is not None:
                assert self._pin is not None
                self._pin.__exit__(None, None, None)
                self._pin, self._artifacts = None, None
            self.closed = True
