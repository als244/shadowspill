"""Creation conversion uses real archives/publication; CPU state is not a worker."""

import hashlib
import io
import os
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch

from engine_tests.checkpoint.test_conversion import (
    FIRST,
    GRADIENT,
    SECOND,
    WEIGHT,
    HostRank,
    adam,
    family,
)
from engine_tests.offline._example import communication, options, profiled
from engine_tests.server.test_artifacts import reference, store
from engine_tests.server.test_checkpoint_restore import configuration
from engine_tests.server.test_checkpoints import stream_rank
from shadowspill import MemoryBudgets, plan_program
from shadowspill.checkpoint import checkpoint_size, load_checkpoint
from shadowspill.checkpoint.identity import state_identity
from shadowspill.checkpoint.preparation import plan_states
from shadowspill.orchestration.launch.admission import (
    NodeAdmission,
    NodeCapacity,
    ResourceRequest,
)
from shadowspill.orchestration.launch.log_files import LogFiles
from shadowspill.orchestration.launch.processes import worker_command
from shadowspill.server import checkpoint_conversion as module
from shadowspill.server.checkpoint_restore import CheckpointResolver
from shadowspill.server.checkpoints import publish_checkpoint
from shadowspill.training.preparation.bundle import TrainingBundle
from shadowspill.training.worker.checkpoint_records import RankSave
from shadowspill.training.worker.conversion_records import ConversionCommand
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.checkpoints import CheckpointInfo
from shadowspill_client.protocol.codec import encode
from shadowspill_client.protocol.requests import ArtifactRef


def bundle(config, count, dim, master, *, missing_default=False):
    base = plan_program(
        profiled(),
        memory_budgets=MemoryBudgets(100, 64),
        communication_info=communication(),
        search_options=options(),
    )
    plans = {}
    for state in family(count, dim, master=master):
        program = state.plan.program
        if missing_default:
            program = replace(
                program,
                bindings={
                    **program.bindings,
                    "optimizer_bindings": [
                        {**row, "initialization": {"kind": "external"}}
                        if row["role"] == "hyperparameter"
                        else row
                        for row in program.bindings["optimizer_bindings"]
                    ],
                },
            )
        saved = replace(
            base,
            profiled_program=replace(base.profiled_program, program=program),
            _costed_program=replace(
                base._costed_program,
                devices=program.devices,
                alias_groups=program.alias_groups,
                objects=program.objects,
                tasks=(),
            ),
            memory_layout=None,
            _admission_facts=None,
            pool_memory=None,
        )
        plans[state.logical_rank] = {program.digest: saved}
    return TrainingBundle(config, {}, plans, {})


@contextmanager
def configured(
    tmp_path,
    *,
    source_count=2,
    target_count=4,
    weights="master",
    master=True,
    missing_default=False,
):
    config = configuration()
    source = bundle(config, source_count, 0, master)
    target = bundle(config, target_count, 1, master, missing_default=missing_default)
    memory, disk = BufferBudget(64 << 20), BufferBudget(8 << 20)
    with store(tmp_path) as artifacts:
        body = encode(config)
        config_ref = reference(body, "config")
        artifacts.put(config_ref, lambda _: io.BytesIO(body))
        plan_refs = []
        for name, saved in (("source-plans", source), ("target-plans", target)):
            path = tmp_path / (name + ".ssp")
            saved.save(path)
            ref = ArtifactRef(
                name, hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size
            )
            artifacts.put(ref, lambda _, path=path: path.open("rb"))
            plan_refs.append(ref)
        info = CheckpointInfo(
            "session",
            "source-run",
            "saved-history",
            "save",
            17,
            source_count,
            config_ref,
            plan_refs[0],
            sys.byteorder,
            weights=weights,
            metadata={"cursor": [9, 11]},
        )
        executors = [
            HostRank(state, initialize=True) for state in plan_states(source.plans)
        ]
        ranks = [
            RankSave(
                state.logical_rank,
                checkpoint_size(state, metadata=info.rank_metadata(), weights=weights),
                state_identity(state),
                state.plan.program.digest,
                {},
            )
            for state in executors
        ]
        checkpoint = publish_checkpoint(
            artifacts,
            info,
            ranks,
            lambda rank, sink: stream_rank(executors[rank.rank], info, sink),
        )
        resolved = CheckpointResolver(
            artifacts, session_id="session", world_size=None, configuration=config
        ).resolve(checkpoint)
        kwargs = dict(
            plans=plan_refs[1],
            configuration=config_ref,
            config=config,
            root=tmp_path,
            memory=memory,
            disk=disk,
            max_disk_bytes=8 << 20,
            max_files=100,
            world_size=target_count,
            host_bytes=2 << 30,
        )
        yield artifacts, resolved, target.plans, kwargs
        assert memory.used == disk.used == (0, 0)
        assert not list(tmp_path.glob(".checkpoint-conversion-*"))


def execute(artifacts, source, kwargs, *, cancel=False):
    node = NodeAdmission(NodeCapacity("cpu", (), 4 << 30))
    with TemporaryDirectory(dir=kwargs["root"]) as directory:
        root = Path(directory)
        lease = node.reserve(ResourceRequest({}, kwargs["host_bytes"]))
        logs = LogFiles(
            lease,
            None,
            root=root,
            disk=kwargs["disk"],
            memory=kwargs["memory"],
            limit_bytes=4096,
        )
        try:
            child = lease.launch_host(
                lambda fd: worker_command(
                    "shadowspill.training.worker.convert_checkpoint", (fd,)
                ),
                environment=os.environ,
                directory=logs.directory,
                cwd=root,
                log_limit_bytes=4096,
            )
            assert not node.used.devices
            if cancel:
                child.close()
            return module.prepare_checkpoint(artifacts, source, child, **kwargs)
        finally:
            lease.close()
            logs.close()
            assert node.used.claims == 0


@pytest.mark.parametrize("counts", [(1, 4), (4, 1), (2, 4), (4, 2)])
@pytest.mark.parametrize(
    "weights,master", [("master", True), ("compute", True), ("master", False)]
)
def test_creation_converts_publishes_reuses_and_preserves_next_adam_update(
    tmp_path, monkeypatch, counts, weights, master
):
    with configured(
        tmp_path,
        source_count=counts[0],
        target_count=counts[1],
        weights=weights,
        master=master,
    ) as (artifacts, source, target, kwargs):
        result = execute(artifacts, source, kwargs)
        info = result.manifest.info
        assert info.conversion.source == source.reference
        assert info.conversion.source_plans == source.manifest.info.plans
        assert info.plans == kwargs["plans"] and info.world_size == counts[1]
        assert info.completed_updates == 17 and info.weights == weights
        assert info.history_id == source.manifest.info.history_id
        expected_weight = (
            WEIGHT if weights == "master" or not master else WEIGHT.bfloat16().float()
        )
        expected, _ = adam(expected_weight, FIRST, SECOND, GRADIENT, 17)
        for state, row in zip(plan_states(target), result.manifest.ranks, strict=True):
            path = tmp_path / f"converted-{row.rank}.ssp"
            with artifacts.open(row.archive) as stream:
                path.write_bytes(stream.read())
            restored = HostRank(state)
            assert (
                load_checkpoint(restored, path, chunk_bytes=17) == info.rank_metadata()
            )
            selected = slice(
                (counts[1] - row.rank - 1) * (12 // counts[1]),
                (counts[1] - row.rank) * (12 // counts[1]),
            )
            shape = (4, 12 // counts[1])
            name = "master.linear.weight" if master else "linear.weight"
            actual, state = adam(
                restored.values[name].reshape(shape),
                restored.values["optimizer.linear.weight.exp_avg"].reshape(shape),
                restored.values["optimizer.linear.weight.exp_avg_sq"].reshape(shape),
                GRADIENT[:, selected],
                restored.values["optimizer.linear.weight.step"].item(),
            )
            torch.testing.assert_close(
                actual, expected[:, selected], rtol=1e-6, atol=1e-7
            )
            assert state["step"].item() == 18
            assert restored.values["optimizer_group.0.lr"].item() == 0
        send = module.write_message

        def forbid_writes(stream, message):
            assert (
                not isinstance(message, ConversionCommand) or message.action != "write"
            )
            return send(stream, message)

        monkeypatch.setattr(module, "write_message", forbid_writes)
        before = artifacts.used
        assert execute(artifacts, source, kwargs) == result
        assert artifacts.used == before


@pytest.mark.parametrize(
    "fault", ["disk", "files", "host", "publication", "default", "cancel"]
)
def test_conversion_failure_publishes_no_target_generation_and_releases_resources(
    tmp_path, monkeypatch, fault
):
    with configured(tmp_path, missing_default=fault == "default") as (
        artifacts,
        source,
        _target,
        kwargs,
    ):
        if fault == "disk":
            kwargs["max_disk_bytes"] = 1
        elif fault == "files":
            kwargs["max_files"] = 1
        elif fault == "host":
            kwargs["host_bytes"] = 1
        elif fault == "publication":
            original = module.publish_checkpoint
            writes = []

            def fail_last(artifacts, info, ranks, write):
                def fail(rank, sink):
                    writes.append(rank.rank)
                    if rank.rank == 3:
                        raise OSError("last rank failed")
                    return write(rank, sink)

                return original(artifacts, info, ranks, fail)

            monkeypatch.setattr(module, "publish_checkpoint", fail_last)
        before = artifacts.used
        with pytest.raises((ValueError, OSError, RuntimeError)):
            execute(artifacts, source, kwargs, cancel=fault == "cancel")
        assert (
            artifacts.used == before
            and artifacts.get(source.reference.id) == source.reference
        )
        assert kwargs["memory"].used == (0, 0)
        if fault == "publication":
            assert writes == [0, 1, 2, 3]


def test_live_restore_still_rejects_worker_count_changes(tmp_path):
    with configured(tmp_path) as (artifacts, source, target, kwargs):
        resolver = CheckpointResolver(
            artifacts,
            session_id="session",
            world_size=len(target),
            configuration=kwargs["config"],
        )
        with pytest.raises(ValueError, match="ownership requires"):
            resolver.resolve(source.reference)
