"""The best plan a search has found so far, shared by whoever is searching.

PressureFit plans one resolved program. A Program has several, and a plan
admitted under any one of them is a real plan -- so it can bound the search
under every other. This is the object that carries that bound across the
boundary, and its scope is the caller's decision:

- one instance threaded from resolved program to resolved program gives an
  ordered, deterministic search where each one starts from the last one's
  answer;
- one instance per resolved program gives independent searches that share
  nothing, which is what parallelism without cross-program pruning looks
  like;
- one instance shared by concurrent searches prunes hardest, because every
  worker sees an admission the moment it happens -- at the cost of a result
  that depends on which worker arrived first.

That last mode is genuinely non-deterministic and must never be the one a
gate or a corpus replay runs in. `deterministic` records which was asked
for, so a caller can refuse to accept a plan built the fast way.
"""

from __future__ import annotations

import threading


class BestFound:
    """A makespan to beat, readable and updatable by a running search."""

    def __init__(self, *, deterministic: bool = True) -> None:
        self._lock = threading.Lock()
        self._makespan_ns: int | None = None
        self.deterministic = deterministic

    @property
    def makespan_ns(self) -> int | None:
        """The best makespan admitted so far, or None if nothing has been."""

        with self._lock:
            return self._makespan_ns

    @property
    def bound_ns(self) -> int:
        """The bound in the form the planner takes: zero means no bound."""

        with self._lock:
            return 0 if self._makespan_ns is None else self._makespan_ns

    def offer(self, makespan_ns: int) -> bool:
        """Record `makespan_ns` if it beats what is held. Returns whether it did.

        Ties do not replace the incumbent, so a search that revisits an
        equally good plan does not perturb what is already held.
        """

        if makespan_ns <= 0:
            raise ValueError("a makespan offered to BestFound must be positive")
        with self._lock:
            if self._makespan_ns is not None and makespan_ns >= self._makespan_ns:
                return False
            self._makespan_ns = makespan_ns
            return True


__all__ = ["BestFound"]
