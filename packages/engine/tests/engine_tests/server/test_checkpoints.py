"""Real rank archives use all-rank publication without a second state serializer."""

import hashlib
import io
import socket
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
import torch

from engine_tests.checkpoint.test_weights import mixed_executor
from engine_tests.execution.test_checkpoint import MemoryExecutor
from engine_tests.execution.test_partial_checkpoint import executor as partial_executor
from engine_tests.server.test_artifacts import reference, store
from shadowspill.checkpoint import checkpoint_size, load_checkpoint, write_checkpoint
from shadowspill.checkpoint.archive import WriteAll, read_manifest
from shadowspill.checkpoint.identity import state_identity
from shadowspill.server.checkpoints import (
    publish_checkpoint,
    stage_checkpoint,
)
from shadowspill.training.worker.checkpoint_records import (
    RankSave,
    RankSaved,
    checkpoint_id,
)
from shadowspill_client.protocol.checkpoints import CheckpointInfo, CheckpointManifest
from shadowspill_client.protocol.codec import decode


def boundary(artifacts, *, count=2, request="save", weights="master"):
    config, plans = (
        reference(b"configuration", "configuration"),
        reference(b"plans", "plans"),
    )
    artifacts.put(config, lambda _: io.BytesIO(b"configuration"))
    artifacts.put(plans, lambda _: io.BytesIO(b"plans"))
    return CheckpointInfo(
        "session",
        "run",
        "history",
        request,
        9,
        count,
        config,
        plans,
        sys.byteorder,
        weights=weights,
        metadata={"cursor": 17},
    )


def workers(info):
    executors, ranks = [], []
    for rank in range(info.world_size):
        executor = MemoryExecutor()
        executor.logical_rank = rank
        executor.plan.program.digest = "a" * 64
        executor.values["step"].fill_(info.completed_updates)
        executor.values["bf16"].add_(rank)
        executors.append(executor)
        ranks.append(
            RankSave(
                rank,
                checkpoint_size(
                    executor, metadata=info.rank_metadata(), weights=info.weights
                ),
                state_identity(executor),
                executor.plan.program.digest,
                {},
            )
        )
    return executors, ranks


def stream_rank(executor, info, destination, *, fault=None):
    """A bounded actual socket transfer; completion is separate from stream EOF."""
    send, receive = socket.socketpair()
    send.settimeout(5)
    receive.settimeout(5)

    class Sink:
        def __init__(self):
            self.digest = hashlib.sha256()

        def write(self, data):
            count = send.send(data)
            self.digest.update(data[:count])
            return count

    def export():
        try:
            sink = Sink()
            manifest = write_checkpoint(
                executor,
                sink,
                metadata=info.rank_metadata(),
                chunk_bytes=16,
                weights=info.weights,
                frozen_state_id=info.frozen_state_id,
            )
            if fault is not None:
                fault(manifest)
            return RankSaved(sink.digest.hexdigest(), manifest)
        finally:
            send.shutdown(socket.SHUT_WR)

    with send, receive, ThreadPoolExecutor(1) as pool:
        pending = pool.submit(export)
        while block := receive.recv(64 << 10):
            WriteAll(destination).write(block)
        return pending.result(5)


@pytest.mark.parametrize("weights", ["master", "compute"])
def test_complete_socket_generation_reopens_and_restores_every_rank(tmp_path, weights):
    with store(tmp_path) as artifacts:
        info = boundary(artifacts, weights=weights)
        executors, ranks = workers(info)
        expected = [
            {name: value.clone() for name, value in executor.values.items()}
            for executor in executors
        ]
        saved = publish_checkpoint(
            artifacts,
            info,
            ranks,
            lambda rank, sink: stream_rank(executors[rank.rank], info, sink),
            name="latest",
        )
        assert saved.id == checkpoint_id(info)
        assert artifacts.resolve_alias("session", "run", "latest") == saved
        assert artifacts.used[1] == 5  # Configuration, Plans, two ranks, manifest.
    with store(tmp_path) as artifacts:
        with artifacts.open(saved) as stream:
            manifest = decode(stream.read())
        assert isinstance(manifest, CheckpointManifest) and manifest.info == info
        assert len(manifest.ranks) == 2
        for rank in manifest.ranks:
            path = tmp_path / f"restore-{rank.rank}.ssp"
            with artifacts.open(rank.archive) as stream, path.open("wb") as output:
                while block := stream.read(64 << 10):
                    output.write(block)
            executor = executors[rank.rank]
            for value in executor.values.values():
                value.zero_()
            assert (
                load_checkpoint(executor, path, chunk_bytes=24) == info.rank_metadata()
            )
            for name, value in executor.values.items():
                torch.testing.assert_close(
                    value.float(), expected[rank.rank][name].float()
                )
            assert rank.state_sha256 == state_identity(executor)


@pytest.mark.parametrize(
    "field",
    [
        "logical_rank",
        "state",
        "program",
        "additional_programs",
        "metadata",
        "weights",
        "frozen_state_id",
        "byteorder",
        "version",
        "format",
        "completion",
    ],
)
def test_rank_identity_or_completion_failure_leaves_previous_generation_usable(
    tmp_path, field
):
    with store(tmp_path) as artifacts:
        info = boundary(artifacts)
        executors, ranks = workers(info)
        saved = publish_checkpoint(
            artifacts,
            info,
            ranks,
            lambda rank, sink: stream_rank(executors[rank.rank], info, sink),
            name="latest",
        )
        before = artifacts.used
        next_info = replace(info, request_id="next")
        _, next_ranks = workers(next_info)

        def damage(manifest):
            if field == "completion":
                raise OSError("rank died before confirmation")
            manifest[field] = "wrong"

        def write(rank, sink):
            return stream_rank(
                executors[rank.rank],
                next_info,
                sink,
                fault=damage if rank.rank == 1 else None,
            )

        with pytest.raises((ValueError, OSError), match=r"admission|confirmation"):
            publish_checkpoint(artifacts, next_info, next_ranks, write, name="latest")
        assert artifacts.used == before
        assert artifacts.get(checkpoint_id(next_info)) is None
        assert artifacts.resolve_alias("session", "run", "latest") == saved
        assert not list(artifacts._scratch.iterdir())


@pytest.mark.parametrize(
    "failure", ["missing_rank", "duplicate_rank", "dependency", "size", "metadata"]
)
def test_invalid_or_unadmitted_generation_never_starts_a_rank(tmp_path, failure):
    with store(tmp_path) as artifacts:
        info = boundary(artifacts)
        _, ranks = workers(info)
        if failure == "missing_rank":
            ranks = ranks[:1]
        elif failure == "duplicate_rank":
            ranks = [ranks[0], ranks[0]]
        elif failure == "dependency":
            info = replace(info, plans=replace(info.plans, id="foreign-plans"))
        elif failure == "size":
            ranks = [replace(rank, size_bytes=1 << 30) for rank in ranks]
        else:
            info = replace(info, metadata={"too_big": "x" * (1 << 20)})

        def never(*args):
            pytest.fail("rank export started before complete admission")

        before = artifacts.used
        with pytest.raises(ValueError):
            publish_checkpoint(artifacts, info, ranks, never)
        assert artifacts.used == before


def test_coordinator_and_generation_records_import_without_frameworks():
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, *args):
        if name.split('.')[0] in {'torch', 'triton', 'pynvml', 'shadowspill_workloads'}:
            raise AssertionError('framework in checkpoint coordinator: ' + name)
sys.meta_path.insert(0, Block())
from shadowspill.server.checkpoints import publish_checkpoint
from shadowspill.training.worker.checkpoint_records import RankSave, RankSaved
from shadowspill_client.protocol.checkpoints import CheckpointManifest
assert 'shadowspill.checkpoint.save' not in sys.modules
assert 'shadowspill.checkpoint.load' not in sys.modules
""",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("weights", ["master", "compute"])
def test_selected_weights_and_momentum_continue_a_real_cpu_optimizer(tmp_path, weights):
    executor = mixed_executor()
    executor.plan.program.digest = "a" * 64
    master = torch.nn.Parameter(executor.values["master.weight"])
    optimizer = torch.optim.SGD([master], lr=0.1, momentum=0.9)
    optimizer.state[master]["momentum_buffer"] = executor.values["moment"]
    initial = (
        executor.values["master.weight"]
        if weights == "master"
        else executor.values["weight"].float()
    )
    reference_master = torch.nn.Parameter(initial.clone())
    reference_optimizer = torch.optim.SGD([reference_master], lr=0.1, momentum=0.9)
    reference_optimizer.state[reference_master]["momentum_buffer"] = executor.values[
        "moment"
    ].clone()
    with store(tmp_path) as artifacts:
        info = boundary(artifacts, count=1, weights=weights)
        rank = RankSave(
            0,
            checkpoint_size(executor, metadata=info.rank_metadata(), weights=weights),
            state_identity(executor),
            executor.plan.program.digest,
            {},
        )
        saved = publish_checkpoint(
            artifacts,
            info,
            [rank],
            lambda _, sink: stream_rank(executor, info, sink),
        )
        with artifacts.open(saved) as stream:
            manifest = decode(stream.read())
        path = tmp_path / "resume.ssp"
        with (
            artifacts.open(manifest.ranks[0].archive) as source,
            path.open("wb") as dest,
        ):
            while block := source.read(64 << 10):
                dest.write(block)
    with tarfile.open(path, "r:") as archive:
        written = read_manifest(archive)
    selected = "master.weight" if weights == "master" else "weight"
    assert {entry["name"] for entry in written["tensors"]} == {selected, "moment"}
    for value in executor.values.values():
        value.zero_()
    load_checkpoint(executor, path, chunk_bytes=7)
    master.grad = torch.ones_like(master)
    reference_master.grad = torch.ones_like(reference_master)
    optimizer.step()
    reference_optimizer.step()
    torch.testing.assert_close(master, reference_master, rtol=0, atol=0)
    torch.testing.assert_close(
        optimizer.state[master]["momentum_buffer"],
        reference_optimizer.state[reference_master]["momentum_buffer"],
        rtol=0,
        atol=0,
    )


def test_frozen_base_identity_is_retained_without_copying_its_payload(tmp_path):
    executor = partial_executor()
    executor.plan.program.digest = "a" * 64
    with store(tmp_path) as artifacts:
        info = replace(boundary(artifacts, count=1), frozen_state_id="base-v1")
        rank = RankSave(
            0,
            checkpoint_size(
                executor, metadata=info.rank_metadata(), frozen_state_id="base-v1"
            ),
            state_identity(executor),
            executor.plan.program.digest,
            {},
        )
        saved = publish_checkpoint(
            artifacts,
            info,
            [rank],
            lambda _, sink: stream_rank(executor, info, sink),
        )
        with artifacts.open(saved) as stream:
            manifest = decode(stream.read())
        assert manifest.info.frozen_state_id == "base-v1"
        with (
            artifacts.open(manifest.ranks[0].archive) as stream,
            tarfile.open(fileobj=stream, mode="r:") as archive,
        ):
            written = read_manifest(archive)
            assert written["omitted_parameters"] == ["base", "tied_base"]
            assert "base" not in {entry["name"] for entry in written["tensors"]}


def test_completed_checkpoint_can_wait_for_receipt_or_be_discarded(tmp_path):
    with store(tmp_path) as artifacts:
        info = boundary(artifacts)
        executors, ranks = workers(info)
        before = artifacts.used
        with stage_checkpoint(
            artifacts,
            info,
            ranks,
            lambda rank, sink: stream_rank(executors[rank.rank], info, sink),
            name="latest",
        ) as pending:
            assert pending.references[-1].id == checkpoint_id(info)
            assert artifacts.get(checkpoint_id(info)) is None
            assert artifacts.resolve_alias("session", "run", "latest") is None
            assert artifacts.used[1] == before[1] + 3
        assert artifacts.used == before
        assert artifacts.get(checkpoint_id(info)) is None
        assert not list(artifacts._scratch.iterdir())
