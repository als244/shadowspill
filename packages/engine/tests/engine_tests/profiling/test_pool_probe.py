"""Pool/lane lifecycle observations need no installed PyTorch allocator."""

from pathlib import Path

import pytest

from shadowspill.libraries import resolve_library
from shadowspill.memory import PinnedHostPool
from shadowspill.profiling.pool_probe import PoolProbe
from shadowspill.ssd import SSDPool


def _mock() -> Path:
    path = resolve_library("libshadowspill_backend_mock.so")
    assert path is not None
    return path


@pytest.mark.parametrize("kind", ["pinned", "ssd"])
def test_exact_capacity_routes_release_and_reuse(tmp_path, kind):
    spill = (
        PinnedHostPool(32 << 20)
        if kind == "pinned"
        else SSDPool(
            32 << 20,
            directory=tmp_path,
            staging_bytes=1 << 20,
            chunk_bytes=65536,
            queue_depth=2,
        )
    )
    with PoolProbe(spill, 0, backend_library=_mock()) as probe:
        before = probe.snapshot()["backend"]
        for _ in range(2):
            probe.acquire()
            ready = probe.snapshot()["backend"]
            assert (
                ready["bytes_device_allocated"] - before["bytes_device_allocated"]
                == 4 << 20
            )
            registered = (
                ready["bytes_pinned_host_registered"]
                - before["bytes_pinned_host_registered"]
            )
            if kind == "pinned":
                assert registered == spill.capacity
            else:
                assert 0 < registered < spill.capacity
            with pytest.raises(RuntimeError, match="already owns"):
                probe.acquire()
            probe.warmup()
            warmed = probe.snapshot()["backend"]
            assert warmed["copies_host_to_device"] > before["copies_host_to_device"]
            assert warmed["copies_device_to_host"] > before["copies_device_to_host"]
            probe.release()
            probe.release()
            before = probe.snapshot()["backend"]
            for created, freed in (
                ("bytes_device_allocated", "bytes_device_freed"),
                ("bytes_pinned_host_registered", "bytes_pinned_host_unregistered"),
                ("streams_created", "streams_destroyed"),
                ("events_created", "events_destroyed"),
            ):
                assert before[created] == before[freed]
    probe.close()
    with pytest.raises(RuntimeError, match="closed"):
        probe.snapshot()
    assert not list(tmp_path.iterdir())


def test_failed_acquisition_can_be_observed_and_closed():
    with PoolProbe(
        PinnedHostPool(32 << 20, kind=233), 0, backend_library=_mock()
    ) as probe:
        with pytest.raises(RuntimeError, match="acquire failed"):
            probe.acquire()
        stats = probe.snapshot()["backend"]
        assert stats["bytes_device_allocated"] == stats["bytes_device_freed"]
        assert (
            stats["bytes_pinned_host_registered"]
            == stats["bytes_pinned_host_unregistered"]
        )
        with pytest.raises(RuntimeError, match="no acquired pools"):
            probe.warmup()


@pytest.mark.parametrize("value", [0, -1, True, 1 << 64])
def test_capacity_must_not_wrap_at_the_native_boundary(value):
    with pytest.raises(ValueError, match="uint64"):
        PoolProbe(PinnedHostPool(32 << 20), 0, execution_bytes=value)


def test_missing_entrypoint_does_not_open_a_session(monkeypatch):
    from shadowspill.profiling import pool_probe

    monkeypatch.setattr(pool_probe.ct, "CDLL", lambda *_: object())
    with pytest.raises(RuntimeError, match="rebuild"):
        PoolProbe(PinnedHostPool(32 << 20), 0, backend_library=_mock())
