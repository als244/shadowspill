"""Bounded service ownership under concurrent I/O, eviction and failed cleanup."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Event

import pytest

from shadowspill.server.owners import OwnerCache
from shadowspill.server.records.requests import RequestRecords, RunIdentity


@dataclass
class Owner:
    name: str
    closes: int = 0
    fail: bool = False

    def close(self):
        self.closes += 1
        if self.fail:
            raise OSError("cleanup failed")


def test_live_borrows_share_an_owner_and_only_idle_lru_entries_are_evicted():
    cache = OwnerCache(Owner, max_owners=2)
    with cache.use("a") as a, cache.use("a") as again:
        assert again is a and cache.used == (1, 2)
        with cache.use("b") as b:
            assert cache.used == (2, 3)
            with pytest.raises(TimeoutError), cache.use("c"):
                pass
        with cache.use("c") as c:
            assert b.closes == 1 and a.closes == 0
        with cache.use("a"):
            pass
    with cache.use("d") as d:
        assert c.closes == 1 and a.closes == 0 and cache.used == (2, 1)
    cache.close()
    cache.close()
    assert a.closes == d.closes == 1 and cache.used == (0, 0) and cache.closed


def test_opening_is_charged_and_does_not_block_unrelated_existing_owners():
    started, release = Event(), Event()

    def factory(key):
        if key == "slow":
            started.set()
            assert release.wait(5)
        return Owner(key)

    cache = OwnerCache(factory, max_owners=2)
    with cache.use("pinned") as pinned, ThreadPoolExecutor(1) as workers:
        future = workers.submit(lambda: use_once(cache, "slow"))
        try:
            assert started.wait(5) and cache.used == (2, 1)
            with cache.use("pinned") as again:
                assert again is pinned
            with pytest.raises(TimeoutError), cache.use("slow"):
                pass
            with pytest.raises(TimeoutError), cache.use("full"):
                pass
        finally:
            release.set()
        future.result(5)
    cache.close()


def use_once(cache, key):
    with cache.use(key) as value:
        return value


def test_failed_construction_releases_its_slot():
    def factory(key):
        if key == "bad":
            raise ValueError("invalid journal")
        return Owner(key)

    cache = OwnerCache(factory, max_owners=1)
    with pytest.raises(ValueError), cache.use("bad"):
        pass
    assert cache.used == (0, 0)
    use_once(cache, "good")
    cache.close()


@pytest.mark.parametrize("replacement", ["a", "b"])
def test_failed_retirement_is_charged_and_never_borrowed_again(replacement):
    cache = OwnerCache(Owner, max_owners=1)
    old = use_once(cache, "a")
    old.fail = True
    with pytest.raises(OSError), cache.use("b"):
        pass
    assert old.closes == 1 and cache.used == (1, 0)
    with pytest.raises(OSError), cache.use(replacement):
        pass
    old.fail = False
    assert use_once(cache, replacement) is not old
    assert old.closes == 3 and cache.used == (1, 0)
    cache.close()


def test_shutdown_attempts_all_idle_owners_and_preserves_live_borrows():
    cache = OwnerCache(Owner, max_owners=3)
    failed = use_once(cache, "failed")
    good = use_once(cache, "good")
    failed.fail = True
    with cache.use("pinned") as pinned:
        with pytest.raises(ExceptionGroup, match="cleanup"):
            cache.close()
        assert good.closes == 1 and failed.closes == 1 and pinned.closes == 0
        assert cache.used == (2, 1) and not cache.closed
        with pytest.raises(RuntimeError), cache.use("pinned"):
            pass
        failed.fail = False
        with pytest.raises(ExceptionGroup):
            cache.close()
        assert failed.closes == 2 and cache.used == (1, 1)
    cache.close()
    assert pinned.closes == 1 and cache.closed


@pytest.mark.parametrize("fail", [False, True])
def test_shutdown_during_construction_retires_the_value_without_handing_it_out(fail):
    started, release = Event(), Event()
    owner = Owner("late", fail=fail)

    def factory(_):
        started.set()
        assert release.wait(5)
        return owner

    cache = OwnerCache(factory, max_owners=1)
    with ThreadPoolExecutor(1) as workers:
        future = workers.submit(lambda: use_once(cache, "late"))
        try:
            assert started.wait(5)
            with pytest.raises(ExceptionGroup):
                cache.close()
            assert cache.used == (1, 0) and not cache.closed
        finally:
            release.set()
        with pytest.raises(OSError if fail else RuntimeError):
            future.result(5)
    assert owner.closes == 1
    if fail:
        assert cache.used == (1, 0) and not cache.closed
        owner.fail = False
        cache.close()
        assert owner.closes == 2
    assert cache.closed and cache.used == (0, 0)


def test_close_callback_does_not_block_other_borrows_or_close_twice():
    started, release = Event(), Event()

    class Slow(Owner):
        def close(self):
            self.closes += 1
            if self.name == "slow":
                started.set()
                assert release.wait(5)

    cache = OwnerCache(Slow, max_owners=2)
    slow = use_once(cache, "slow")
    with cache.use("pinned") as pinned, ThreadPoolExecutor(1) as workers:
        future = workers.submit(lambda: use_once(cache, "new"))
        try:
            assert started.wait(5)
            with cache.use("pinned") as again:
                assert again is pinned
            with pytest.raises(TimeoutError), cache.use("slow"):
                pass
            with pytest.raises(ExceptionGroup):
                cache.close()
        finally:
            release.set()
        with pytest.raises(RuntimeError):
            future.result(5)
    cache.close()
    assert slow.closes == pinned.closes == 1 and cache.used == (0, 0)


def test_evicted_request_records_reopen_with_original_identity_and_writer(tmp_path):
    def factory(key):
        return RequestRecords(
            tmp_path / key,
            identity=RunIdentity("owner", "session", key),
            cache_root=tmp_path,
        )

    cache = OwnerCache(factory, max_owners=1)
    with cache.use("a") as records:
        lease = records.claim_writer("writer", "server")
        with pytest.raises(RuntimeError, match="active writer"):
            factory("a")
    use_once(cache, "b")
    with cache.use("a") as reopened:
        assert reopened is not records and reopened.writer == lease
    cache.close()
    assert not list(tmp_path.glob("request-index-*"))


def test_request_cleanup_failure_retains_exclusive_owner_for_retry(
    tmp_path, monkeypatch
):
    def factory(key):
        return RequestRecords(
            tmp_path / key,
            identity=RunIdentity("owner", "session", key),
            cache_root=tmp_path,
        )

    cache = OwnerCache(factory, max_owners=1)
    records = use_once(cache, "a")
    cleanup = records._index._directory.cleanup

    def fail():
        raise OSError("index deletion failed")

    monkeypatch.setattr(records._index._directory, "cleanup", fail)
    with pytest.raises(OSError), cache.use("b"):
        pass
    assert cache.used == (1, 0)
    with pytest.raises(RuntimeError, match="unavailable"):
        records.get("request")
    with pytest.raises(RuntimeError, match="active writer"):
        factory("a")
    monkeypatch.setattr(records._index._directory, "cleanup", cleanup)
    use_once(cache, "b")
    cache.close()
    with factory("a"):
        pass
