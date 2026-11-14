"""Durable prefixes survive partial writes; uncertain publication is never retried."""

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from shadowspill.server.records import Journal, JournalLimits
from shadowspill.server.records import journal as storage


def test_grouped_records_roundtrip_owned_data_and_incremental_positions(tmp_path):
    root = tmp_path / "journal"
    source = {"request": "r1", "details": [1, 2]}
    with Journal(root) as journal:
        records = journal.append(
            [("accepted", source), ("dispatch", {"request": "r1"})]
        )
        source["details"].append(3)
        assert records[0].data["details"] == (1, 2)
        assert [item.position for item in records] == [1, 2]
        with pytest.raises(TypeError):
            records[0].data["request"] = "changed"
        journal.append([("completed", {"request": "r1", "update": 1})])
        assert journal.position == 3
    with Journal(root) as reopened:
        assert reopened.position == 3
        assert [item.kind for item in reopened.records()] == [
            "accepted",
            "dispatch",
            "completed",
        ]
        assert reopened.append([("accepted", {"request": "r2"})])[0].position == 4


def test_recovery_removes_only_unpublished_tails_and_keeps_other_files(tmp_path):
    with Journal(tmp_path) as journal:
        journal.append([("accepted", {"id": "r1"})])
    (tmp_path / ".ssj.pending-torn").write_bytes(b"half-written record")
    (tmp_path / "keep.txt").write_text("unrelated")
    with Journal(tmp_path) as journal:
        assert journal.position == 1
        assert not (tmp_path / ".ssj.pending-torn").exists()
        assert (tmp_path / "keep.txt").read_text() == "unrelated"


@pytest.mark.parametrize("damage", ["checksum", "truncate", "gap", "reorder"])
def test_committed_corruption_fails_without_repairing_or_losing_evidence(
    tmp_path, damage
):
    with Journal(tmp_path) as journal:
        journal.append([("accepted", {"id": "r1"})])
        journal.append([("completed", {"id": "r1"})])
    first, second = sorted(tmp_path.glob("*.ssj"))
    if damage == "checksum":
        value = second.read_bytes()
        second.write_bytes(value[:-2] + b"xx")
    elif damage == "truncate":
        second.write_bytes(second.read_bytes()[:17])
    elif damage == "gap":
        first.unlink()
    else:
        a, b = first.read_bytes(), second.read_bytes()
        first.write_bytes(b)
        second.write_bytes(a)
    before = {p.name: p.read_bytes() for p in tmp_path.glob("*.ssj")}
    with pytest.raises(ValueError):
        Journal(tmp_path)
    assert {p.name: p.read_bytes() for p in tmp_path.glob("*.ssj")} == before


@pytest.mark.parametrize("stage", ["file_sync", "publish", "directory_sync"])
def test_storage_failure_poisoning_and_reopen_reconciliation(
    tmp_path, monkeypatch, stage
):
    journal = Journal(tmp_path)
    journal.append([("accepted", {"id": "r1"})])

    def fail(*args, **kwargs):
        raise OSError("injected storage failure")

    with monkeypatch.context() as patch:
        if stage == "file_sync":
            patch.setattr(storage.os, "fsync", fail)
        elif stage == "publish":
            patch.setattr(storage.os, "link", fail)
        else:
            patch.setattr(storage, "_sync_directory", fail)
        with pytest.raises(OSError, match="injected"):
            journal.append([("dispatch", {"id": "r1"})])
        with pytest.raises(RuntimeError, match="durability failed"):
            journal.append([("dispatch", {"id": "r1"})])
    journal.close()
    with Journal(tmp_path) as recovered:
        # A failed directory sync may have published bytes: inspection sees the
        # original dispatch, never an implicit second attempt at the mutation.
        expected = 2 if stage == "directory_sync" else 1
        assert recovered.position == expected
        assert len(list(recovered.records())) == expected


def test_writer_lock_is_exclusive_across_processes_and_released_on_close(tmp_path):
    code = """
import sys
from shadowspill.server.records import Journal
try:
    Journal(sys.argv[1]).close()
except RuntimeError as error:
    assert 'active writer' in str(error)
    sys.exit(7)
"""
    with Journal(tmp_path):
        result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], timeout=15)
        assert result.returncode == 7
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], timeout=15)
    assert result.returncode == 0


def test_limits_reject_before_publication_and_do_not_poison_writer(tmp_path):
    limits = JournalLimits(record_bytes=80, commit_bytes=400, commit_records=2)
    with Journal(tmp_path, limits=limits) as journal:
        for entries in ([], [("event", {})] * 3, [("event", {"big": "x" * 100})]):
            with pytest.raises(ValueError):
                journal.append(entries)
            assert journal.position == 0 and not list(tmp_path.glob("*.ssj"))
        assert journal.append([("event", {})])[0].position == 1
    with pytest.raises(ValueError, match="byte limit"):
        Journal(tmp_path, limits=JournalLimits(record_bytes=80, commit_bytes=100))


def test_concurrent_producers_share_one_contiguous_committed_order(tmp_path):
    with Journal(tmp_path) as journal, ThreadPoolExecutor(max_workers=4) as producers:
        records = list(
            producers.map(lambda i: journal.append([("event", {"i": i})])[0], range(20))
        )
        assert sorted(item.position for item in records) == list(range(1, 21))
        assert {item.data["i"] for item in journal.records()} == set(range(20))


def test_commit_is_exclusive_and_cannot_replace_an_existing_file(tmp_path):
    with Journal(tmp_path) as journal:
        path = tmp_path / "00000000000000000001.ssj"
        path.write_bytes(b"previous evidence")
        with pytest.raises(FileExistsError):
            journal.append([("event", {})])
        assert path.read_bytes() == b"previous evidence"


def test_killed_writer_releases_lock_and_retains_committed_prefix(tmp_path):
    code = """
import os, sys
from shadowspill.server.records import Journal
journal = Journal(sys.argv[1])
journal.append([('dispatch', {'request_id': 'r1'})])
os._exit(9)
"""
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], timeout=15)
    assert result.returncode == 9
    with Journal(tmp_path) as journal:
        assert next(journal.records()).kind == "dispatch"


def test_records_import_without_frameworks_or_native_runtime(tmp_path):
    code = """
import sys
for name in ('torch', 'triton', 'mlops', 'shadowspill_workloads',
             'shadowspill.runtime'):
    sys.modules[name] = None
from shadowspill.server.records import Journal
with Journal(sys.argv[1]) as journal:
    assert journal.append([('accepted', {'request_id': 'r1'})])[0].position == 1
"""
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], timeout=15)
    assert result.returncode == 0


@pytest.mark.parametrize("published", [False, True])
def test_process_death_at_publication_recovers_exactly_the_visible_prefix(
    tmp_path, published
):
    with Journal(tmp_path) as journal:
        journal.append([("accepted", {"request_id": "r1"})])
    code = """
import os, sys
from shadowspill.server.records import Journal
from shadowspill.server.records import journal as storage
journal = Journal(sys.argv[1])
original = storage.os.link
def die(source, destination):
    if sys.argv[2] == 'True':
        original(source, destination)
    os._exit(9)
storage.os.link = die
journal.append([('dispatch', {'request_id': 'r1'})])
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), str(published)], timeout=15
    )
    assert result.returncode == 9
    assert list(tmp_path.glob(".ssj.pending-*"))
    with Journal(tmp_path) as recovered:
        assert [row.kind for row in recovered.records()] == (
            ["accepted", "dispatch"] if published else ["accepted"]
        )
        assert not list(tmp_path.glob(".ssj.pending-*"))
