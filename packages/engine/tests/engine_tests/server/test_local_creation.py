"""Actual phase/worker process ownership with CPU sentinel Plans, not GPU proof."""

import hashlib
import io
import os
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from engine_tests.server.test_jobs import until
from engine_tests.server.test_live import CHILD
from engine_tests.server.test_storage import IDENTITY
from engine_tests.training.test_preparation_templates import inventory
from engine_tests.training.test_setup_resolution import installed
from engine_tests.training.test_worker_startup import prepared_worker
from shadowspill.orchestration.launch.admission import (
    AssignedDevice,
    NodeAdmission,
    NodeCapacity,
    ResourceRequest,
)
from shadowspill.orchestration.launch.host_arena import HostArena
from shadowspill.orchestration.launch.phases import PhaseInvocation
from shadowspill.parallelism import InputFieldPlacement, InputPlacement, Parallelism
from shadowspill.server import local_creation as module
from shadowspill.server import plan_summaries
from shadowspill.server.creation_records import CreationRecord
from shadowspill.server.jobs import RunJobs
from shadowspill.server.local_creation import LocalCreationFactory, LocalCreationLimits
from shadowspill.server.storage import ServiceStorage
from shadowspill.training.preparation.job import BUNDLE_NAME, JOB_NAME, PreparationJob
from shadowspill.training.worker import launch as launch_module
from shadowspill.training.worker.resources import WorkerHost
from shadowspill_client import Weights
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.creation import TrainingCreation
from shadowspill_client.protocol.requests import (
    ArtifactRef,
    OperationRequest,
    RequestIdentity,
)


@contextmanager
def configured(
    tmp_path,
    monkeypatch,
    *,
    change=lambda c: c,
    saved=False,
    fault=None,
    limits_change=lambda limits: limits,
    ranks=1,
    artifact_count=100,
):
    inv = inventory()
    if ranks > 1:
        fields = [
            InputFieldPlacement(tuple(field.path))
            for field in inv.layouts.setup.config.inputs.fields
        ]
        inv = inventory(
            parallelism=Parallelism.dp(
                ranks_per_replica=ranks,
                num_replicas=1,
                inputs=InputPlacement.independent(ranks, fields=fields),
            )
        )
    family = prepared_worker(inv).bindings.prepared
    config = change(family.inventory.layouts.setup.config)
    family = replace(
        family,
        inventory=replace(
            family.inventory,
            layouts=replace(
                family.inventory.layouts,
                setup=replace(family.inventory.layouts.setup, config=config),
            ),
        ),
    )
    node = NodeAdmission(
        NodeCapacity(
            "local",
            tuple(
                AssignedDevice("gpu" if rank == 0 else f"gpu{rank}", str(rank), 4 << 30)
                for rank in range(ranks)
            ),
            16 << 30,
        )
    )
    disk = BufferBudget(128 << 20, 100)
    phases, owners = [], []
    monkeypatch.setattr(module, "load_registry", lambda _: installed())
    monkeypatch.setattr(
        launch_module,
        "worker_command",
        lambda module, args: (
            [sys.executable, "-c", CHILD, str(args[1])]
            if ranks == 1
            else [
                sys.executable,
                str(Path(__file__).with_name("_distributed_creation_child.py")),
                str(args[1]),
            ]
        ),
    )
    payload = b"CPU sentinel bundle; GPU fixture uses actual compiled Plans"
    reference = ArtifactRef(
        "saved-plans", hashlib.sha256(payload).hexdigest(), len(payload)
    )
    bundle = SimpleNamespace(config=config, recheck=lambda _: family)
    monkeypatch.setattr(module.TrainingBundle, "load", lambda _: bundle)
    # These are CPU sentinel worker Plans, without real simulation/layout evidence.
    # Report attribution is tested separately against actual saved Plan artifacts.
    monkeypatch.setattr(
        plan_summaries,
        "occupancy_data",
        lambda plan, *, description: {
            "plan": description,
            "view": "simulated",
            "clock": "CPU service fixture",
            "end_seconds": 1,
            "pools": [],
            "compute": [],
            "fetch": [],
            "evict": [],
        },
    )

    def prepare(root, *, packages, config, environment, launch):
        assert not saved
        root.mkdir()
        PreparationJob(tuple(packages), config).save(root / JOB_NAME)
        for phase in ("build", "profile", "communicate", "plan"):
            assert not node.used.devices
            phases.append(phase)
            result = launch(
                PhaseInvocation(
                    phase,
                    (sys.executable, "-c", f"print({phase!r})"),
                    environment,
                    1,
                    root,
                    root / (phase + ".log"),
                )
            )
            assert result.returncode == 0 and not node.used.devices
        (root / BUNDLE_NAME).write_bytes(payload)
        return bundle

    monkeypatch.setattr(module, "prepare_job", prepare)
    with (
        HostArena(node, 160 << 20, 8) as store_arena,
        HostArena(node, 128 << 20, 1) as job_arena,
        ServiceStorage(
            tmp_path / "service",
            cache_root=tmp_path / "cache",
            incarnation="server",
            memory=store_arena.budget,
            request_host_bytes=128 << 20,
            max_request_owners=1,
            max_artifact_owners=1,
            artifact_bytes=32 << 20,
            artifact_count=artifact_count,
        ) as service,
    ):
        service.recover()
        limits = LocalCreationLimits(
            64 << 20,
            64 << 20,
            WorkerHost(64 << 20, 64 << 20),
            32 << 20,
            32 << 20,
            64 << 10,
            100,
        )
        limits = limits_change(limits)
        factory = LocalCreationFactory(
            service,
            node,
            packages=(),
            root=tmp_path,
            cache=tmp_path / "cache",
            environment=os.environ,
            limits=limits,
            disk=disk,
            base_host_bytes=288 << 20,
        )

        def construct(info):
            before = node.used
            owner = factory(info)
            assert before == node.used and owner._directory is None
            owners.append(owner)
            if fault is not None:
                fault(owner)
            return owner

        jobs = RunJobs(
            service,
            construct,
            memory=job_arena.budget,
            host_bytes=128 << 20,
            max_runs=1,
            reap_interval=0.01,
        )
        jobs.start()
        ctx = SimpleNamespace(
            factory=factory,
            jobs=jobs,
            node=node,
            disk=disk,
            service=service,
            owners=owners,
            phases=phases,
            config=config,
            saved=reference if saved else None,
            bundle=bundle,
            job_memory=job_arena.budget,
        )
        if saved:
            with service.artifacts("owner") as artifacts:
                artifacts.put(reference, lambda _: io.BytesIO(payload))
        try:
            yield ctx
        finally:
            jobs.close(timeout=5)
            assert all(owner.closed for owner in owners)
            assert node.used.host_bytes == 288 << 20
            assert disk.used == job_arena.budget.used == (0, 0)
    assert node.used.claims == 0 and store_arena.budget.used == (0, 0)


def submit(ctx, **options):
    creation = TrainingCreation(ctx.config, plans=ctx.saved, **options)
    record = CreationRecord.encode(creation)

    def publish():
        with ctx.service.artifacts("owner") as artifacts:
            record.publish(artifacts)

    return ctx.jobs.create(
        IDENTITY,
        request_sha256=record.reference.sha256,
        configuration=record.configuration,
        publish=publish,
    )


@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("ranks", [1, 2])
def test_factory_owns_preparation_and_persistent_child_without_a_recipe_loop(
    tmp_path, monkeypatch, saved, ranks
):
    with configured(tmp_path, monkeypatch, saved=saved, ranks=ranks) as ctx:
        submit(ctx)
        until(lambda: ctx.service.catalog.get(IDENTITY).state != "preparing")
        assert ctx.service.catalog.get(IDENTITY).state == "ready", ctx.jobs.failures()
        until(lambda: next(iter(ctx.jobs._jobs.values())).live)
        owner = ctx.owners[0]
        assert all(
            worker.startup.logical_by_worker == tuple(range(ranks))
            for worker in owner._node.prepared.values()
        )
        assert ctx.phases == (
            [] if saved else ["build", "profile", "communicate", "plan"]
        )
        assert owner._phases is None  # All temporary process/log owners retired.
        controller = owner.run.controller
        lease = controller.attach("writer").lease
        close = OperationRequest(
            RequestIdentity("close", "run", 1, "writer", lease.writer_epoch, "server"),
            "close",
        )
        controller.register(close)
        controller.receive("close")
        assert controller.wait("close", timeout=5).state == "completed"
        until(lambda: ctx.jobs.used == (0, 0))
        assert owner.closed and owner.run.closed and owner._arena is None
        assert ctx.service.catalog.get(IDENTITY).state == "closed"
        if not saved:
            with ctx.service.artifacts("owner") as artifacts:
                names = [ref.id for ref in artifacts.page()]
                assert any(name.startswith("prepared.") for name in names)
                assert (
                    len([name for name in names if name.startswith("preparation.")])
                    == 6
                )


@pytest.mark.parametrize(
    "change",
    [
        lambda c: replace(c, model=replace(c.model, initialize=None)),
        lambda c: replace(
            c, model=replace(c.model, initialize=Weights.artifact("missing"))
        ),
        lambda c: replace(c, execution=replace(c.execution, nsys={"steps": [1]})),
    ],
)
def test_unavailable_startup_choices_fail_before_any_phase(
    tmp_path, monkeypatch, change
):
    with configured(tmp_path, monkeypatch, change=change) as ctx:
        submit(ctx)
        until(lambda: ctx.jobs.used == (0, 0))
        assert ctx.service.catalog.get(IDENTITY).state == "failed"
        assert not ctx.phases and ctx.owners[0]._directory is None


def test_checkpoint_is_not_silently_replaced_with_fresh_weights(tmp_path, monkeypatch):
    with configured(tmp_path, monkeypatch) as ctx:
        submit(ctx, checkpoint=ArtifactRef("checkpoint", "a" * 64, 100))
        until(lambda: ctx.jobs.used == (0, 0))
        assert "checkpoint is not" in ctx.service.catalog.get(IDENTITY).error.message
        assert not ctx.phases


def test_waiting_job_stays_queued_until_phase_admission_and_stop_preserves_other_lease(
    tmp_path, monkeypatch
):
    with configured(tmp_path, monkeypatch) as ctx:
        occupied = ctx.node.reserve(ResourceRequest({"gpu": 4 << 30}, 0))
        try:
            submit(ctx, wait_for_resources=True)
            until(lambda: ctx.owners and ctx.owners[0]._phases is not None)
            assert ctx.service.catalog.get(IDENTITY).stage == "queued"
            ctx.jobs.close(timeout=5)
            assert not occupied.closed and ctx.node.used.devices == {"gpu"}
            assert ctx.service.catalog.get(IDENTITY).state == "stopped"
        finally:
            occupied.close()


def test_impossible_combined_phase_host_demand_never_waits_or_launches(
    tmp_path, monkeypatch
):
    with configured(tmp_path, monkeypatch) as ctx:
        ctx.factory.limits = replace(
            ctx.factory.limits, preparation_host_bytes=16 << 30
        )
        submit(ctx, wait_for_resources=True)
        until(lambda: ctx.jobs.used == (0, 0))
        assert "cannot fit" in ctx.service.catalog.get(IDENTITY).error.message
        assert not ctx.phases


def test_failed_preparation_publication_keeps_artifacts_and_retries_cleanup_only(
    tmp_path, monkeypatch
):
    release = Event()

    def fault(owner):
        original = owner._publish_metadata

        def publish(*args):
            if not release.is_set():
                raise OSError("preparation archive unavailable")
            return original(*args)

        monkeypatch.setattr(owner, "_publish_metadata", publish)

    with configured(tmp_path, monkeypatch, fault=fault) as ctx:
        try:
            submit(ctx)
            until(ctx.jobs.failures)
            assert ctx.owners[0].run is None and ctx.owners[0]._directory is not None
            assert ctx.disk.used[0] > 0 and ctx.job_memory.used == (128 << 20, 1)
            assert ctx.phases == ["build", "profile", "communicate", "plan"]
        finally:
            release.set()
        until(lambda: ctx.jobs.used == (0, 0))
        assert ctx.service.catalog.get(IDENTITY).state == "failed"
        assert len(ctx.phases) == 4  # Cleanup never restarts preparation or execution.
        with ctx.service.artifacts("owner") as artifacts:
            assert any(ref.id.startswith("preparation.") for ref in artifacts.page())
