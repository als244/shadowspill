"""Agent delivery admits storage before copying and owns it through consumption."""

import hashlib
import io
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event

import pytest

from shadowspill.orchestration.launch.staging import COPY_BYTES, FileDelivery
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.codec import ProtocolError
from shadowspill_client.protocol.requests import ArtifactRef


def references(values):
    return tuple(
        ArtifactRef(str(i), hashlib.sha256(value).hexdigest(), len(value))
        for i, value in enumerate(values)
    )


def test_delivery_is_owned_bounded_and_kept_until_every_consumer_finishes(tmp_path):
    values = [b"abc" * COPY_BYTES, b"", b"final"]
    refs = references(values)
    disk, memory = BufferBudget(sum(map(len, values))), BufferBudget(1 << 20)
    requests = []

    class Source(io.BytesIO):
        def read(self, size=-1):
            assert 0 < size <= COPY_BYTES
            requests.append(size)
            return super().read(min(size, 7000))

    with FileDelivery(refs, root=tmp_path, disk=disk, memory=memory) as delivered:
        before = memory.used[0]
        delivered.receive(lambda reference: Source(values[int(reference.id)]))
        assert delivered.ready and memory.used[0] == before - COPY_BYTES
        assert disk.used == (sum(map(len, values)), 1)
        with delivered.use() as first:
            assert [first(ref).read_bytes() for ref in refs] == values
            paths = [first(ref) for ref in refs]
            with pytest.raises(ValueError, match="not owned"):
                first(replace(refs[0], sha256="0" * 64))
            with pytest.raises(RuntimeError, match="consumers"):
                delivered.close()
            with delivered.use() as second:
                assert second(refs[0]) == first(refs[0])
        with delivered.use() as second:
            with pytest.raises(RuntimeError, match="no longer borrowed"):
                first(refs[0])
            assert second(refs[0]).is_file()
    assert disk.used == memory.used == (0, 0)
    assert all(not path.exists() for path in paths) and not list(tmp_path.iterdir())
    assert requests and max(requests) <= COPY_BYTES


@pytest.mark.parametrize("resource", ["disk", "memory"])
def test_quotas_are_reserved_before_creating_files_or_opening_sources(
    tmp_path, resource
):
    refs = references([b"data"])
    disk = BufferBudget(1 if resource == "disk" else 1 << 20)
    memory = BufferBudget(1 if resource == "memory" else 1 << 20)
    with pytest.raises(ValueError, match="limit"):
        FileDelivery(refs, root=tmp_path, disk=disk, memory=memory)
    assert disk.used == memory.used == (0, 0)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("bad", [b"shor", b"longer", b"other"])
def test_later_corruption_exposes_no_ready_files_and_cleans_earlier_copies(
    tmp_path, bad
):
    refs = references([b"first", b"value"])
    disk, memory = BufferBudget(10), BufferBudget(1 << 20)
    with FileDelivery(refs, root=tmp_path, disk=disk, memory=memory) as delivered:
        with pytest.raises(ProtocolError):
            delivered.receive(
                lambda ref: io.BytesIO(b"first" if ref.id == "0" else bad)
            )
        assert not delivered.ready
        with pytest.raises(RuntimeError, match="not ready"), delivered.use():
            pass
        with pytest.raises(RuntimeError, match="already started"):
            delivered.receive(lambda ref: io.BytesIO(b"value"))
    assert disk.used == memory.used == (0, 0)
    assert not list(tmp_path.iterdir())


def test_rejected_metadata_and_failed_directory_creation_release_all_reservations(
    tmp_path,
):
    refs = references([b"data"])
    disk, memory = BufferBudget(100), BufferBudget(1 << 20)
    for selected, root, error in (
        ((*refs, *refs), tmp_path, ValueError),
        (refs, tmp_path / "absent", FileNotFoundError),
        ([replace(refs[0], media_type="x" * (1 << 20))], tmp_path, ProtocolError),
    ):
        with pytest.raises(error):
            FileDelivery(selected, root=root, disk=disk, memory=memory)
        assert disk.used == memory.used == (0, 0)
    assert not list(tmp_path.iterdir())


def test_cancellation_keeps_active_producer_ownership_until_its_read_returns(tmp_path):
    entered, release = Event(), Event()
    refs = references([b"payload"])
    disk, memory = BufferBudget(7), BufferBudget(1 << 20)

    class Blocked(io.BytesIO):
        def read(self, size=-1):
            entered.set()
            assert release.wait(5)
            return super().read(size)

    with FileDelivery(refs, root=tmp_path, disk=disk, memory=memory) as delivered:
        before = memory.used
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(delivered.receive, lambda ref: Blocked(b"payload"))
            assert entered.wait(5)
            delivered.cancel()
            with pytest.raises(RuntimeError, match="producer"):
                delivered.close()
            assert disk.used == (7, 1) and memory.used == before
            release.set()
            with pytest.raises(InterruptedError, match="cancelled") as failure:
                future.result(timeout=5)
        assert memory.used[0] == before[0] - COPY_BYTES
        traceback = failure.value.__traceback__
        while traceback:
            if traceback.tb_frame.f_code.co_name == "_copy":
                assert "block" not in traceback.tb_frame.f_locals
                assert "part" not in traceback.tb_frame.f_locals
            traceback = traceback.tb_next
        assert not delivered.ready
    assert disk.used == memory.used == (0, 0)


def test_failed_deletion_keeps_quotas_until_cleanup_succeeds(tmp_path, monkeypatch):
    refs = references([b"payload"])
    disk, memory = BufferBudget(7), BufferBudget(1 << 20)
    delivered = FileDelivery(refs, root=tmp_path, disk=disk, memory=memory)
    delivered.receive(lambda ref: io.BytesIO(b"payload"))

    def fail():
        raise OSError("temporary cleanup failure")

    with monkeypatch.context() as patch:
        patch.setattr(delivered._directory, "cleanup", fail)
        with pytest.raises(OSError, match="cleanup"):
            delivered.close()
        assert disk.used == (7, 1) and memory.used[0] > 0 and not delivered.closed
    delivered.close()
    assert disk.used == memory.used == (0, 0)


def test_real_prepared_frames_survive_sender_cleanup_and_worker_integrity_checks(
    tmp_path,
):
    from engine_tests.training.test_worker_commands import invocation

    with invocation(tmp_path) as (data, _, command, bindings, sources):
        total = sum(ref.size_bytes for ref in command.frames)
        disk, memory = BufferBudget(total), BufferBudget(1 << 20)
        with FileDelivery(
            command.frames, root=tmp_path, disk=disk, memory=memory
        ) as delivered:
            delivered.receive(lambda ref: sources[ref.id].open("rb", buffering=0))
            data.spool.close()
            assert all(not path.exists() for path in sources.values())
            with delivered.use() as resolve:
                bound = bindings.bind(command, resolve)
                budget = BufferBudget(bound.input_bytes)
                bound.verify_inputs(budget)
                assert budget.used == (0, 0)
        assert disk.used == memory.used == (0, 0)
