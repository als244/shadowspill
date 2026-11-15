"""A rank set, its manifest and its friendly name become visible together."""

import io
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event

import pytest

from engine_tests.server.test_artifact_writer import spec
from engine_tests.server.test_artifacts import reference, store
from shadowspill.server.artifact_writer import ProducedArtifacts
from shadowspill.server.artifacts import INDEX_BYTES, MAX_GROUP_ARTIFACTS
from shadowspill.server.records.artifact_index import ArtifactAlias
from shadowspill_client.buffers import BufferBudget


def generation(prefix="save", count=3):
    return tuple(
        reference(f"payload-{i}".encode(), f"{prefix}.{i}") for i in range(count)
    )


def publish(artifacts, refs, *, alias=None, inspect=None):
    def write(output, writer):
        index = next(i for i, ref in enumerate(refs) if ref.id == output.id)
        writer.write(memoryview(f"payload-{index}".encode()))
        if inspect is not None:
            inspect(index)
        return refs[index]

    return artifacts.produce_group(tuple(map(spec, refs)), write, alias=alias)


def test_members_and_name_stay_private_until_the_complete_generation_commits(tmp_path):
    refs = generation(count=5)  # More ranks than the two concurrent transfer slots.
    alias = ArtifactAlias("session", "run", "latest", refs[-1].id)
    entered, release = Event(), Event()
    memory = BufferBudget(8 << 20, 10)

    def inspect(index):
        if index == 3:
            entered.set()
            assert release.wait(5)

    with store(tmp_path, memory=memory) as artifacts, ThreadPoolExecutor(1) as pool:
        future = pool.submit(publish, artifacts, refs, alias=alias, inspect=inspect)
        try:
            assert entered.wait(5)
            assert not artifacts.page()
            assert artifacts.resolve_alias("session", "run", "latest") is None
            assert artifacts.used == (sum(r.size_bytes for r in refs), len(refs))
            assert artifacts._disk.used == (sum(r.size_bytes for r in refs), 1)
            with pytest.raises(RuntimeError, match="in progress"):
                artifacts.put(refs[2], lambda _: io.BytesIO(b"payload-2"))
            with pytest.raises(RuntimeError, match="producers"):
                artifacts.close()
        finally:
            release.set()
        assert future.result(timeout=5) == refs
        assert artifacts.page() == refs
        assert artifacts.resolve_alias("session", "run", "latest") == refs[-1]
        assert artifacts._disk.used == (0, 0)
        assert memory.used == (INDEX_BYTES, 1)
    with store(tmp_path) as recovered:
        assert recovered.resolve_alias("session", "run", "latest") == refs[-1]
        for i, ref in enumerate(refs):
            with recovered.open(ref) as stream:
                assert stream.read() == f"payload-{i}".encode()
    assert memory.used == (0, 0)


@pytest.mark.parametrize("failed_member", [0, 1, 2])
def test_rank_failure_reclaims_all_partials_and_keeps_the_previous_alias(
    tmp_path, failed_member
):
    first, second = generation("first"), generation("second")
    alias = ArtifactAlias("session", "run", "latest", first[-1].id)

    def fail(index):
        if index == failed_member:
            raise OSError("rank confirmation failed")

    with store(tmp_path) as artifacts:
        publish(artifacts, first, alias=alias)
        with pytest.raises(OSError, match="rank confirmation"):
            publish(
                artifacts,
                second,
                alias=replace(alias, artifact_id=second[-1].id),
                inspect=fail,
            )
        assert artifacts.page() == first
        assert artifacts.used == (sum(r.size_bytes for r in first), len(first))
        assert artifacts.resolve_alias("session", "run", "latest") == first[-1]
        assert not list(artifacts._scratch.iterdir())
        assert artifacts._disk.used == (0, 0)


@pytest.mark.parametrize("failure", ["before_commit", "after_commit", "index", "alias"])
def test_lost_commit_reply_recovers_all_members_and_name_or_none(
    tmp_path, monkeypatch, failure
):
    first, second = generation("first"), generation("second")
    alias = ArtifactAlias("session", "run", "latest", first[-1].id)
    artifacts = store(tmp_path)
    publish(artifacts, first, alias=alias)
    append, insert = artifacts._journal.append, artifacts._index.insert

    def failed_reply(entries):
        if failure == "after_commit":
            append(entries)
        raise OSError("commit reply lost")

    def failed_index(ref):
        if ref == second[1]:
            raise OSError("index failed after one member")
        insert(ref)

    def failed_alias(_):
        raise OSError("alias index failed")

    if failure == "index":
        monkeypatch.setattr(artifacts._index, "insert", failed_index)
    elif failure == "alias":
        monkeypatch.setattr(artifacts._index, "set_alias", failed_alias)
    else:
        monkeypatch.setattr(artifacts._journal, "append", failed_reply)
    with pytest.raises(OSError):
        publish(artifacts, second, alias=replace(alias, artifact_id=second[-1].id))
    with pytest.raises(RuntimeError, match="reconcile"):
        artifacts.resolve_alias("session", "run", "latest")
    artifacts.close()
    with store(tmp_path) as recovered:
        visible = first if failure == "before_commit" else first + second
        assert recovered.page() == visible
        assert recovered.resolve_alias("session", "run", "latest") == visible[-1]
        assert len(list(recovered._data.iterdir())) == len(visible)
        assert not list(recovered._scratch.iterdir())


@pytest.mark.parametrize("limit", ["bytes", "count"])
def test_whole_group_capacity_is_reserved_before_any_encoder_runs(tmp_path, limit):
    refs = generation()

    def never(*args):
        pytest.fail("unadmitted encoder")

    kwargs = {"max_bytes": 20} if limit == "bytes" else {"max_artifacts": 2}
    with store(tmp_path, **kwargs) as artifacts:
        with pytest.raises(ValueError, match="capacity"):
            artifacts.produce_group(tuple(map(spec, refs)), never)
        assert artifacts.used == (0, 0) and not artifacts._pending


def test_group_envelope_conflicts_and_unknown_alias_target_fail_before_writing(
    tmp_path,
):
    refs = generation()

    def never(*args):
        pytest.fail("unadmitted encoder")

    with store(tmp_path) as artifacts:
        for bad in (
            (),
            (spec(refs[0]),) * 2,
            (spec(refs[0]),) * (MAX_GROUP_ARTIFACTS + 1),
        ):
            with pytest.raises(ValueError):
                artifacts.produce_group(bad, never)
        with pytest.raises(ValueError, match="member"):
            artifacts.produce_group(
                tuple(map(spec, refs)),
                never,
                alias=ArtifactAlias("session", "run", "latest", "missing"),
            )
        publish(artifacts, refs)
        with pytest.raises(ValueError, match="already committed"):
            artifacts.produce_group(tuple(map(spec, refs)), never)


def test_names_are_scoped_and_replacing_one_preserves_old_immutable_generations(
    tmp_path,
):
    first, second = generation("first"), generation("second")
    alias = ArtifactAlias("session", "run", "latest", first[-1].id)
    with store(tmp_path) as artifacts:
        publish(artifacts, first, alias=alias)
        assert artifacts.resolve_alias("other", "run", "latest") is None
        assert artifacts.resolve_alias("session", "other", "latest") is None
        publish(artifacts, second, alias=replace(alias, artifact_id=second[-1].id))
        assert artifacts.resolve_alias("session", "run", "latest") == second[-1]
        assert artifacts.get(first[-1].id) == first[-1]
    with store(tmp_path) as recovered:
        assert recovered.page() == first + second
        assert recovered.resolve_alias("session", "run", "latest") == second[-1]


def test_cleanup_failure_keeps_one_group_owner_for_explicit_retry(
    tmp_path, monkeypatch
):
    refs, memory = generation(), BufferBudget(8 << 20, 10)
    artifacts = store(tmp_path, memory=memory)
    close = ProducedArtifacts.close

    def failed_cleanup(_):
        raise OSError("pending storage busy")

    monkeypatch.setattr(ProducedArtifacts, "close", failed_cleanup)
    with pytest.raises(OSError):
        publish(artifacts, refs)
    assert len(artifacts._pending) == 1 and artifacts._disk.used[1] == 1
    with pytest.raises(OSError):
        artifacts.close()
    monkeypatch.setattr(ProducedArtifacts, "close", close)
    artifacts.close()
    assert memory.used == (0, 0)
    with store(tmp_path) as recovered:
        assert recovered.page() == refs


def test_process_loss_after_one_rank_is_not_a_partial_generation(tmp_path):
    child = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import hashlib, os, sys
from pathlib import Path
from shadowspill.server.artifacts import ArtifactStore
from shadowspill.orchestration.launch.publication import OutputSpec
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.requests import ArtifactRef
root = Path(sys.argv[1])
store = ArtifactStore(root/'artifacts', owner_id='owner', cache_root=root,
                      memory=BufferBudget(8 << 20, 10), max_bytes=100, max_artifacts=10)
def write(spec, writer):
    if spec.id == 'rank1':
        os._exit(47)
    writer.write(memoryview(b'rank'))
    return ArtifactRef(spec.id, hashlib.sha256(b'rank').hexdigest(), 4)
store.produce_group([OutputSpec('rank0', 4, 'application/octet-stream'),
                     OutputSpec('rank1', 4, 'application/octet-stream')], write)
""",
            str(tmp_path),
        ],
        capture_output=True,
        timeout=15,
    )
    assert child.returncode == 47, child.stderr
    with store(tmp_path) as recovered:
        assert not recovered.page() and recovered.used == (0, 0)
        assert not list(recovered._scratch.iterdir())
