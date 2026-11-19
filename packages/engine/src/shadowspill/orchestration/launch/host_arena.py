"""Keep bounded node-daemon/artifact memory charged to the node for its full
lifetime."""

from __future__ import annotations

from threading import Event, RLock
from typing import Self

from shadowspill_client.buffers import BufferBudget

from .admission import NodeAdmission, ResourceRequest


class HostPartition:
    """A fixed slice of an already admitted arena, with independent close gates."""

    def __init__(self, parent: HostArena, max_bytes: int, max_requests: int) -> None:
        self.budget = BufferBudget(max_bytes, max_requests)
        self._arena = parent
        self._claim = parent.budget.reserve(max_bytes, timeout=0)
        self._lock = RLock()
        self.closed = self._closing = False

    def require_node(self, node: NodeAdmission) -> None:
        if self._arena._node is not node:
            raise ValueError("preclaimed host memory belongs to another node")
        if self._closing or self._arena._closing:
            raise RuntimeError("preclaimed host arena is closing or closed")

    def close(self) -> None:
        with self._lock:
            if self.closed:
                return
            self._closing = True
            self.budget.close()
            if self.budget.used != (0, 0):
                raise RuntimeError("host partition still has live buffer owners")
            self._claim.close()
            self.closed = True

    def __enter__(self) -> Self:
        if self._closing:
            raise RuntimeError("host partition is closing or closed")
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class HostArena:
    """One node claim backs the complete capacity of a shared buffer budget.

    File delivery, grants, output verification and controller envelopes can use
    separate arenas. Their capacities add even when individual buffers happen to
    be empty. A worker lease cannot release memory still owned by an artifact.
    Disk quotas use their own budgets and are not converted to resident memory.
    """

    def __init__(
        self,
        node: NodeAdmission,
        max_bytes: int,
        max_requests: int,
        *,
        wait: bool = False,
        timeout: float | None = None,
        cancelled: Event | None = None,
    ) -> None:
        self.budget = BufferBudget(max_bytes, max_requests)
        self._node = node
        self._claim = node.reserve(
            ResourceRequest({}, self.budget.max_bytes),
            wait=wait,
            timeout=timeout,
            cancelled=cancelled,
        )
        self._lock = RLock()
        self.closed = self._closing = False

    def partition(self, max_bytes: int, max_requests: int) -> HostPartition:
        """Assign one pre-admitted slice without another node resource wait.

        Partition capacities are charged even when empty. Closing one child's
        budget cannot close a sibling's publication/metadata budget. The complete
        node claim remains until every child partition has actually retired.
        """
        with self._lock:
            if self._closing:
                raise RuntimeError("host arena is closing or closed")
            return HostPartition(self, max_bytes, max_requests)

    def close(self) -> None:
        with self._lock:
            if self.closed:
                return
            self._closing = True
            # Stop new borrows first, waking blocked producers. Existing owners
            # may still finish; a failed close retains the complete node claim.
            self.budget.close()
            if self.budget.used != (0, 0):
                raise RuntimeError("host arena still has live buffer owners")
            self._claim.close()
            self.closed = True

    def __enter__(self) -> Self:
        if self.closed or self._closing:
            raise RuntimeError("host arena is closing or closed")
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
