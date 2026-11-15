"""Creation identity and restart inspection never recreate mutable training state."""

import subprocess
import sys
from dataclasses import replace

import pytest

from shadowspill.server.records.journal import Journal
from shadowspill.server.records.requests import RunIdentity
from shadowspill.server.records.run_index import RUN_BYTES, run_bytes
from shadowspill.server.records.runs import INDEX_BYTES, RunCatalog
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.codec import ProtocolError, decode, encode
from shadowspill_client.protocol.requests import ArtifactRef
from shadowspill_client.protocol.runs import CREATION_STAGES, RunInfo
from shadowspill_client.protocol.service import ServiceError

IDENTITY = RunIdentity("owner", "session", "run")
CONFIG = ArtifactRef("config", "a" * 64, 100)
PLANS = ArtifactRef("plans", "b" * 64, 1000)
ERROR = ServiceError("failed", "Creation could not finish.")


def catalog(root, *, incarnation="server", memory=None):
    return RunCatalog(
        root / "runs",
        incarnation=incarnation,
        cache_root=root,
        memory=memory or BufferBudget(INDEX_BYTES, 1),
    )


def create(records, identity=IDENTITY, **changes):
    return records.create(
        identity,
        **{
            "request_sha256": "c" * 64,
            "configuration": CONFIG,
            "name": "Small classifier",
            "labels": {"experiment": "trial"},
            **changes,
        },
    )


def ready(records, identity=IDENTITY):
    records.progress(identity, "initializing")
    return records.ready(identity, plans=PLANS, initial_history_id="history")


def test_exact_creation_retries_and_original_provenance_survive_close_and_restart(
    tmp_path,
):
    budget = BufferBudget(INDEX_BYTES, 1)
    with catalog(tmp_path, memory=budget) as records:
        original, added = create(records)
        assert added and original.state == "preparing" and original.stage == "queued"
        assert create(records) == (original, False)
        for stage in CREATION_STAGES[1:]:
            progress = records.progress(IDENTITY, stage)
            assert progress.stage == stage
            assert records.progress(IDENTITY, stage) == progress
        initialized = records.ready(IDENTITY, plans=PLANS, initial_history_id="history")
        assert initialized.state == "ready" and create(records) == (initialized, False)
        closed = records.finish(IDENTITY)
        assert closed.state == "closed" and records.finish(IDENTITY) == closed
        assert closed.plans == PLANS and closed.configuration == CONFIG
    assert budget.used == (0, 0) and not list(tmp_path.glob("run-index-*"))
    with catalog(tmp_path, incarnation="replacement", memory=budget) as records:
        assert records.get(IDENTITY) == closed
        assert create(records) == (closed, False)
        assert not records.interrupted()
        with pytest.raises(RuntimeError, match="earlier server"):
            records.ready(IDENTITY, plans=PLANS, initial_history_id="history")
    assert budget.used == (0, 0)


def test_registration_leaves_room_for_maximum_failure_and_ready_provenance(tmp_path):
    error = ServiceError("x" * 128, "\U0010ffff" * 1024)
    plans = ArtifactRef("p" * 128, "b" * 64, (1 << 63) - 1, "x" * 256)
    labels = {str(i): "x" * 256 for i in range(16)}
    with catalog(tmp_path) as records:
        create(records, labels=labels, name="x" * 256)
        records.progress(IDENTITY, "initializing")
        records.ready(IDENTITY, plans=plans, initial_history_id="h" * 128)
        stopped = records.stop(IDENTITY, error)
        assert 8192 < len(run_bytes(stopped)) <= RUN_BYTES
        failed_id = replace(IDENTITY, run_id="failed")
        create(records, failed_id, labels=labels, name="x" * 256)
        failed = records.fail(failed_id, error)
        assert len(run_bytes(failed)) <= RUN_BYTES
    with catalog(tmp_path, incarnation="replacement") as records:
        assert records.get(IDENTITY) == stopped
        assert records.get(failed_id) == failed
        assert not records.interrupted()


@pytest.mark.parametrize(
    "change",
    [
        {"request_sha256": "d" * 64},
        {"configuration": replace(CONFIG, sha256="d" * 64)},
        {"name": "different"},
        {"labels": {"experiment": "another"}},
    ],
)
def test_conflicting_run_ids_cannot_create_a_second_job(tmp_path, change):
    with catalog(tmp_path) as records:
        original, _ = create(records)
        with pytest.raises(ValueError, match="different creation"):
            create(records, **change)
        assert records.get(IDENTITY) == original


def test_owner_session_and_cursor_scopes_do_not_leak_other_runs(tmp_path):
    with catalog(tmp_path) as records:
        for owner in ("owner", "other"):
            for session in ("session", "other"):
                for name in ("c", "a", "b"):
                    create(records, RunIdentity(owner, session, name))
        page = records.page("owner", "session", limit=2)
        assert [info.run_id for info in page] == ["a", "b"]
        assert all(
            info.owner_id == "owner" and info.session_id == "session" for info in page
        )
        assert [
            info.run_id for info in records.page("owner", "session", after="b")
        ] == ["c"]
        assert records.page("unknown", "session") == ()
        assert records.get(IDENTITY) is None
        with pytest.raises(KeyError):
            records.progress(IDENTITY, "build")
        with pytest.raises(ValueError):
            records.page("owner", "../session")
        with pytest.raises(ValueError):
            records.page("owner", "session", limit=129)


def test_restart_exposes_old_unfinished_runs_without_claiming_or_replaying_them(
    tmp_path,
):
    identities = [replace(IDENTITY, run_id=str(index)) for index in range(5)]
    with catalog(tmp_path) as records:
        for identity in identities:
            create(records, identity)
        ready(records, identities[1])
        records.fail(identities[2], ERROR)
        ready(records, identities[3])
        records.finish(identities[3])
        records.stop(identities[4], ERROR)
    with catalog(tmp_path, incarnation="new") as records:
        page = records.interrupted(limit=1)
        assert len(page) == 1 and page[0][1].run_id == "0"
        position = page[0][0]
        stopped = records.stop(
            identities[0], ServiceError("server_restarted", "Creation was interrupted.")
        )
        assert stopped.state == "stopped" and stopped.plans is None
        assert records.stop(identities[0], stopped.error) == stopped
        page = records.interrupted(after=position)
        assert len(page) == 1 and page[0][1].run_id == "1"
        assert page[0][1].plans == PLANS and page[0][1].initial_history_id == "history"
        with pytest.raises(RuntimeError, match="earlier server"):
            records.progress(identities[1], "initializing")
        records.stop(identities[1], ERROR)
        assert not records.interrupted()
        new = replace(IDENTITY, run_id="new")
        create(records, new)
        assert not records.interrupted()
        assert create(records, identities[0]) == (stopped, False)


def test_creation_progress_cannot_restart_and_readiness_requires_initialization(
    tmp_path,
):
    with catalog(tmp_path) as records:
        create(records)
        with pytest.raises(ValueError, match="initialization"):
            records.ready(IDENTITY, plans=PLANS, initial_history_id="history")
        records.progress(IDENTITY, "profile")
        with pytest.raises(ValueError, match="backward"):
            records.progress(IDENTITY, "build")
        # Reused compatible artifacts may skip phases; they cannot restart one.
        initialized = ready(records)
        with pytest.raises(ValueError):
            records.fail(IDENTITY, ERROR)
        assert records.get(IDENTITY) == initialized
        stopped = records.stop(IDENTITY, ERROR)
        assert stopped.plans == initialized.plans
        with pytest.raises(ValueError):
            records.finish(IDENTITY)
        assert records.get(IDENTITY) == stopped


def test_index_failure_after_durable_readiness_recovers_the_original_outcome(
    tmp_path, monkeypatch
):
    records = catalog(tmp_path)
    create(records)
    records.progress(IDENTITY, "initializing")

    def fail(info):
        raise OSError("index unavailable")

    monkeypatch.setattr(records._index, "update", fail)
    with pytest.raises(OSError):
        records.ready(IDENTITY, plans=PLANS, initial_history_id="history")
    with pytest.raises(RuntimeError, match="reconcile"):
        records.get(IDENTITY)
    records.close()
    with catalog(tmp_path, incarnation="new") as recovered:
        info = recovered.get(IDENTITY)
        assert info.state == "ready" and info.plans == PLANS
        assert recovered.interrupted()[0][1] == info


def test_catalog_lock_and_failed_index_cleanup_keep_their_owners(tmp_path, monkeypatch):
    budget = BufferBudget(INDEX_BYTES * 2, 2)
    records = catalog(tmp_path, memory=budget)
    with pytest.raises(RuntimeError, match="active writer"):
        catalog(tmp_path, memory=budget)
    assert budget.used == (INDEX_BYTES, 1)
    cleanup = records._index._directory.cleanup

    def fail():
        raise OSError("index deletion unavailable")

    monkeypatch.setattr(records._index._directory, "cleanup", fail)
    try:
        with pytest.raises(OSError):
            records.close()
        assert not records.closed and budget.used == (INDEX_BYTES, 1)
        with pytest.raises(RuntimeError, match="closing"):
            records.get(IDENTITY)
    finally:
        monkeypatch.setattr(records._index._directory, "cleanup", cleanup)
        records.close()
    assert budget.used == (0, 0)


def test_semantically_corrupt_committed_lifecycle_is_preserved_and_rejected(tmp_path):
    with catalog(tmp_path) as records:
        original, _ = create(records)
    with Journal(tmp_path / "runs", limits=records._journal.limits) as journal:
        # Valid checksums cannot authorize jumping to initialized readiness.
        journal.append(
            [
                (
                    "run.status",
                    replace(
                        original,
                        state="ready",
                        stage=None,
                        plans=PLANS,
                        initial_history_id="history",
                    ).to_dict(),
                )
            ]
        )
    commits = {
        path.name: path.read_bytes() for path in (tmp_path / "runs").glob("*.ssj")
    }
    budget = BufferBudget(INDEX_BYTES, 1)
    with pytest.raises(ValueError, match="initialization"):
        catalog(tmp_path, memory=budget)
    assert budget.used == (0, 0)
    assert commits == {
        path.name: path.read_bytes() for path in (tmp_path / "runs").glob("*.ssj")
    }


def test_large_run_metadata_fails_before_durable_registration(tmp_path):
    with catalog(tmp_path) as records:
        with pytest.raises(ProtocolError, match="exceeds"):
            create(records, configuration=replace(CONFIG, media_type="x" * 9000))
        assert records.get(IDENTITY) is None
        assert create(records)[1]


@pytest.mark.parametrize(
    "changes",
    [
        {"state": "ready", "stage": None},
        {"state": "closed", "stage": None},
        {"stage": None},
        {"stage": "execute"},
        {"error": ERROR},
        {"state": "failed", "stage": None},
        {"labels": {"a": 1}},
        {"labels": {str(i): "x" for i in range(17)}},
        {"request_sha256": "X" * 64},
        {"owner_id": "../owner"},
    ],
)
def test_portable_run_metadata_requires_consistent_lifecycle_and_bounded_identity(
    changes,
):
    info = RunInfo("owner", "session", "run", "server", "c" * 64, CONFIG)
    assert decode(encode(info)) == info
    with pytest.raises((ValueError, TypeError)):
        replace(info, **changes)


def test_catalog_imports_without_training_framework_runtime_or_catalog(tmp_path):
    script = """
import sys
from pathlib import Path
for name in ('torch','triton','mlops','shadowspill.runtime','shadowspill.training',
             'shadowspill_workloads'):
    sys.modules[name] = None
from shadowspill.server.records.runs import RunCatalog, INDEX_BYTES
from shadowspill.server.records.requests import RunIdentity
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.requests import ArtifactRef
root = Path(sys.argv[1])
with RunCatalog(root/'runs', incarnation='new', cache_root=root,
                memory=BufferBudget(INDEX_BYTES, 1)) as records:
    info, created = records.create(RunIdentity('o','s','r'), request_sha256='a'*64,
                                  configuration=ArtifactRef('c','b'*64,1))
    assert created and info.stage == 'queued'
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
