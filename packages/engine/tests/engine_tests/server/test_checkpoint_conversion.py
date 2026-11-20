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
from engine_tests.training.test_worker_randomness import draw
from shadowspill import MemoryBudgets, plan_program
from shadowspill.checkpoint import checkpoint_size, load_checkpoint
from shadowspill.checkpoint.archive import read_manifest
from shadowspill.checkpoint.identity import state_identity
from shadowspill.checkpoint.inspection import open_checkpoint
from shadowspill.checkpoint.preparation import plan_states
from shadowspill.checkpoint.random_streams import (
    RandomState,
    default_streams,
    read_snapshot,
    snapshot,
)
from shadowspill.orchestration.launch.admission import (
    NodeAdmission,
    NodeCapacity,
    ResourceRequest,
)
from shadowspill.orchestration.launch.log_files import LogFiles
from shadowspill.orchestration.launch.processes import worker_command
from shadowspill.pytorch._rng import preserve_rng
from shadowspill.server.checkpoints import publish_checkpoint
from shadowspill.server.startup import conversion as module
from shadowspill.server.startup.checkpoint import CheckpointResolver
from shadowspill.training.preparation.artifacts.bundle import TrainingBundle
from shadowspill.training.worker.checkpoint_records import RankSave
from shadowspill.training.worker.conversion_records import ConversionCommand
from shadowspill.training.worker.randomness import WorkerRandomness
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.checkpoints import SaveInfo
from shadowspill_client.protocol.codec import encode
from shadowspill_client.protocol.frozen import FrozenBase
from shadowspill_client.protocol.requests import ArtifactRef


def bundle(config, count, dim, master, *, missing_default=False, frozen=False):
    base = plan_program(
        profiled(),
        memory_budgets=MemoryBudgets(100, 64),
        communication_info=communication(),
        search_options=options(),
    )
    plans = {}
    for state in family(count, dim, master=master, frozen=frozen):
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
    randomness=False,
    frozen=False,
):
    config = configuration()
    source = bundle(config, source_count, 0, master, frozen=frozen)
    target = bundle(
        config, target_count, 1, master, missing_default=missing_default, frozen=frozen
    )
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
        base = (
            FrozenBase(config.model, config.component_sources["model"])
            if frozen
            else None
        )
        info = SaveInfo(
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
            frozen_base=base,
            frozen_state_id=None if base is None else base.identity,
        )
        executors = [
            HostRank(state, initialize=True) for state in plan_states(source.plans)
        ]
        random_states = {}
        if randomness:
            with preserve_rng(torch.device("cpu")):
                for rank in range(source_count):
                    owner = WorkerRandomness(torch.device("cpu"), rank)
                    seed = 132 if randomness == "seed" and rank == 1 else 131
                    owner.initialize(seed)
                    draw()
                    with owner.evaluation():
                        draw()
                    _, states = read_snapshot(owner.capture())
                    accelerators = [
                        RandomState(stream, seed=rank + 23)
                        for stream in default_streams(
                            rank, torch.__version__, accelerator=True
                        )
                        if stream.identity.component_path[-1] == "accelerator"
                    ]
                    random_states[rank] = snapshot(seed, [*states, *accelerators])
                    if rank == 1:
                        if randomness == "missing":
                            random_states.pop(rank)
                        elif randomness == "rank":
                            random_states[rank]["streams"][0]["stream"]["rank"] = 0
                        elif randomness == "empty":
                            random_states[rank]["streams"] = []

        def metadata(rank):
            return {
                **info.rank_metadata(),
                **(
                    {"randomness": random_states[rank]} if rank in random_states else {}
                ),
            }

        ranks = [
            RankSave(
                state.logical_rank,
                checkpoint_size(
                    state,
                    metadata=metadata(state.logical_rank),
                    weights=weights,
                    frozen_state_id=info.frozen_state_id,
                ),
                state_identity(state),
                state.plan.program.digest,
                {},
                random_states.get(state.logical_rank),
            )
            for state in executors
        ]
        checkpoint = publish_checkpoint(
            artifacts,
            info,
            ranks,
            lambda rank, sink: stream_rank(
                executors[rank.rank], info, sink, metadata=metadata(rank.rank)
            ),
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
            return module.prepare_state(artifacts, source, child, **kwargs)
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


@pytest.mark.parametrize("counts", [(1, 4), (4, 1), (2, 4)])
def test_partial_conversion_preserves_base_omission_exact_sizes_and_next_update(
    tmp_path, counts
):
    with configured(
        tmp_path, source_count=counts[0], target_count=counts[1], frozen=True
    ) as (artifacts, source, target, kwargs):
        result = execute(artifacts, source, kwargs)
        assert result.manifest.info.frozen_base == source.manifest.info.frozen_base
        expected, _ = adam(WEIGHT, FIRST, SECOND, GRADIENT, 17)
        for state, row in zip(plan_states(target), result.manifest.ranks, strict=True):
            path = tmp_path / f"partial-{row.rank}.ssp"
            with artifacts.open(row.archive) as stream:
                path.write_bytes(stream.read())
            assert path.stat().st_size == row.archive.size_bytes
            with open_checkpoint(path) as archive:
                manifest = read_manifest(archive)
            assert manifest["omitted_parameters"] == ["base.tied", "base.weight"]
            selected = slice(
                (counts[1] - row.rank - 1) * (12 // counts[1]),
                (counts[1] - row.rank) * (12 // counts[1]),
            )
            frozen = WEIGHT[:, selected].contiguous()

            def fill(chunk, destination, frozen=frozen):
                assert chunk.name == "base.weight"
                destination.copy_(
                    frozen.reshape(-1)[
                        chunk.offset : chunk.offset + destination.numel()
                    ]
                )

            restored = HostRank(state)
            load_checkpoint(
                restored,
                path,
                chunk_bytes=17,
                frozen_state_id=result.manifest.info.frozen_state_id,
                fill_tensor=fill,
            )
            torch.testing.assert_close(
                restored.values["base.weight"].reshape(frozen.shape),
                frozen,
                rtol=0,
                atol=0,
            )
            actual, _ = adam(
                restored.values["master.linear.weight"].reshape(frozen.shape),
                restored.values["optimizer.linear.weight.exp_avg"].reshape(
                    frozen.shape
                ),
                restored.values["optimizer.linear.weight.exp_avg_sq"].reshape(
                    frozen.shape
                ),
                GRADIENT[:, selected],
                17,
            )
            torch.testing.assert_close(
                actual, expected[:, selected], rtol=1e-6, atol=1e-7
            )


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


@pytest.mark.parametrize("counts", [(2, 4), (4, 2)])
def test_rng_conversion_preserves_mapped_draws_and_records_new_rank_seeds(
    tmp_path, counts
):
    with configured(
        tmp_path, source_count=counts[0], target_count=counts[1], randomness=True
    ) as (artifacts, source, target, kwargs):
        sources = []
        for rank in source.manifest.ranks:
            with (
                artifacts.open(rank.archive) as stream,
                open_checkpoint(stream) as archive,
            ):
                sources.append(read_manifest(archive)["metadata"]["randomness"])
        converted = execute(artifacts, source, kwargs)
        conversion = converted.manifest.info.conversion.randomness
        assert conversion["version"] == 1 and conversion["root_seed"] == 131
        assert len(conversion["source_streams"]) == 8 * counts[0]
        assert len(conversion["target_streams"]) == 8 * counts[1]
        assert conversion[
            "restore_id"
        ] == converted.manifest.info.request_id.removeprefix("conversion-")

        def cpu_snapshot(value):
            seed, streams = read_snapshot(value)
            return snapshot(
                seed,
                [
                    state
                    for state in streams
                    if state.stream.identity.component_path[-1] != "accelerator"
                ],
            )

        with preserve_rng(torch.device("cpu")):
            for state, rank in zip(
                plan_states(target), converted.manifest.ranks, strict=True
            ):
                path = tmp_path / f"converted-rng-{rank.rank}.ssp"
                with artifacts.open(rank.archive) as stream:
                    path.write_bytes(stream.read())
                metadata = load_checkpoint(HostRank(state), path, chunk_bytes=17)
                seed, streams = read_snapshot(metadata["randomness"])
                assert seed == 131
                actions = [
                    row
                    for row in conversion["target_streams"]
                    if row["rank"] == rank.rank
                ]
                if rank.rank < counts[0]:
                    assert set(streams) == set(read_snapshot(sources[rank.rank])[1])
                    assert {row["action"] for row in actions} == {"preserve"}
                    owner = WorkerRandomness(torch.device("cpu"), rank.rank)
                    owner.restore(cpu_snapshot(sources[rank.rank]))
                    expected = draw()
                    with owner.evaluation():
                        evaluated = draw()
                    owner.restore(cpu_snapshot(metadata["randomness"]))
                    assert draw() == expected
                    with owner.evaluation():
                        assert draw() == evaluated
                else:
                    assert {row["action"] for row in actions} == {"derive"}
                    assert all(
                        row.seed is not None and row.payload is None for row in streams
                    )
                    assert {row.seed for row in streams} == {
                        row["seed"] for row in actions
                    }
        before = artifacts.used
        assert execute(artifacts, source, kwargs) == converted
        assert artifacts.used == before


@pytest.mark.parametrize("fault", ["seed", "missing", "rank", "empty"])
def test_invalid_rank_rng_generation_fails_before_any_target_publication(
    tmp_path, fault
):
    with configured(tmp_path, randomness=fault) as (artifacts, source, _, kwargs):
        before = artifacts.used
        with pytest.raises((ValueError, RuntimeError), match="RNG"):
            execute(artifacts, source, kwargs)
        assert (
            artifacts.used == before
            and artifacts.get(source.reference.id) == source.reference
        )
