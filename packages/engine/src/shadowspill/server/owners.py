"""Bounded borrowed record/artifact owners, with retryable idle eviction."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from typing import Literal, Protocol

from shadowspill_client._records import integer


class Closable(Protocol):
    """Close confirms all owned resources were released, or retains them for retry."""

    def close(self) -> None: ...


@dataclass
class _Owner[Value: Closable]:
    value: Value | None = None
    users: int = 0
    state: Literal["opening", "ready", "retiring", "failed"] = "opening"


class OwnerCache[Key: Hashable, Value: Closable]:
    """Bound simultaneous open owners without evicting a live borrow.

    Factories reserve their own complete index/artifact budgets. This cache also
    bounds in-progress opens and failed retirements; it never drops an owner just
    because close failed. Factories and close callbacks run outside the cache lock,
    so a long journal rebuild cannot block unrelated existing owners. Busy opens
    or a full pinned cache fail promptly with backpressure, without another queue.

    Live runs retain a borrow for their lifetime. HTTP lookups retain short borrows
    through actual I/O. The factory must clean up its own failed construction.
    """

    def __init__(self, factory: Callable[[Key], Value], *, max_owners: int) -> None:
        self._factory = factory
        self.max_owners = integer(max_owners, "maximum open owners", minimum=1)
        self._entries: OrderedDict[Key, _Owner[Value]] = OrderedDict()
        self._mutex = RLock()
        self._closing = self.closed = False

    @property
    def used(self) -> tuple[int, int]:
        with self._mutex:
            return len(self._entries), sum(
                owner.users for owner in self._entries.values()
            )

    def _retire(self, key: Key, owner: _Owner[Value]) -> None:
        assert owner.value is not None and owner.state == "retiring" and not owner.users
        try:
            owner.value.close()
        except BaseException:
            with self._mutex:
                owner.state = "failed"
            raise
        with self._mutex:
            assert self._entries[key] is owner
            del self._entries[key]
            if self._closing and not self._entries:
                self.closed = True

    def _load(self, key: Key, owner: _Owner[Value]) -> _Owner[Value]:
        try:
            value = self._factory(key)
        except BaseException:
            with self._mutex:
                del self._entries[key]
                if self._closing and not self._entries:
                    self.closed = True
            raise
        with self._mutex:
            owner.value = value
            if not self._closing:
                owner.state, owner.users = "ready", 1
                return owner
            owner.state = "retiring"
        self._retire(key, owner)
        raise RuntimeError("owner cache closed during construction")

    def _acquire(self, key: Key) -> _Owner[Value]:
        while True:
            with self._mutex:
                if self._closing:
                    raise RuntimeError("owner cache is closing or closed")
                existing = self._entries.get(key)
                if existing is not None and existing.state == "ready":
                    existing.users += 1
                    self._entries.move_to_end(key)
                    return existing
                if existing is not None and existing.state != "failed":
                    raise TimeoutError("this owner is being opened or retired")
                if existing is None and len(self._entries) < self.max_owners:
                    owner: _Owner[Value] = _Owner()
                    self._entries[key] = owner
                    load = True
                else:
                    idle = next(
                        (
                            (item, value)
                            for item, value in self._entries.items()
                            if not value.users
                            and value.state in ("ready", "failed")
                            and (existing is None or value is existing)
                        ),
                        None,
                    )
                    if idle is None:
                        raise TimeoutError(
                            "all cached owners have live users or pending I/O"
                        )
                    retired_key, owner = idle
                    owner.state = "retiring"
                    load = False
            if load:
                return self._load(key, owner)
            self._retire(retired_key, owner)

    @contextmanager
    def use(self, key: Key) -> Iterator[Value]:
        owner = self._acquire(key)
        assert owner.value is not None
        try:
            yield owner.value
        finally:
            with self._mutex:
                owner.users -= 1
                assert owner.users >= 0

    def close(self) -> None:
        with self._mutex:
            self._closing = True
            idle = tuple(
                (key, owner)
                for key, owner in self._entries.items()
                if not owner.users and owner.state in ("ready", "failed")
            )
            for _, owner in idle:
                owner.state = "retiring"
        errors = []
        for key, owner in idle:
            try:
                self._retire(key, owner)
            except BaseException as error:
                errors.append(error)
        with self._mutex:
            self.closed = not self._entries
            if self._entries and not errors:
                errors.append(
                    RuntimeError("cached owners still have live users or pending I/O")
                )
        if errors:
            raise BaseExceptionGroup("cached owners did not finish cleanup", errors)
