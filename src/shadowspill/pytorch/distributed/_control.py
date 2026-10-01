"""Bounded CPU metadata exchange over the process group's rendezvous store.

This channel is for preparation and explicit checkpoint/run-control operations.
Runtime tasks and local fetch/evict never call it.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any, TypeVar, cast

import torch.distributed as dist
from torch._C._distributed_c10d import PrefixStore
from torch.distributed.distributed_c10d import _get_process_group_store

T = TypeVar("T")


class PreparationError(RuntimeError):
    """A participant failed or the preparation contract disagreed."""


class Control:
    def __init__(
        self, group: dist.ProcessGroup, *, namespace: str, timeout: float = 120
    ) -> None:
        if timeout <= 0:
            raise ValueError("preparation timeout must be positive")
        self.group = group
        self.members = tuple(dist.get_process_group_ranks(group))
        self.rank = dist.get_rank()
        if self.rank not in self.members:
            raise ValueError("this rank does not belong to the participant group")
        if not namespace:
            raise ValueError("the control namespace must be nonempty")
        self.store = PrefixStore(
            "shadowspill/" + namespace + "/", _get_process_group_store(group)
        )
        self.timeout = timeout
        self.sequence = 0

    def fail(self, phase: str, error: BaseException) -> None:
        if isinstance(error, PreparationError):
            return
        self.store.set(
            f"failure/{self.rank}",
            json.dumps(
                {
                    "rank": self.rank,
                    "phase": phase,
                    "error": f"{type(error).__name__}: {error}",
                }
            ),
        )

    def check_failure(self) -> None:
        for rank in self.members:
            key = f"failure/{rank}"
            if self.store.check([key]):
                failed = json.loads(self.store.get(key))
                raise PreparationError(
                    f"distributed preparation failed on rank {failed['rank']} "
                    f"during {failed['phase']}: {failed['error']}"
                )

    def exchange(self, phase: str, value: Any) -> tuple[Any, ...]:
        """Exchange one JSON value per participant, detecting phase mismatches."""
        self.check_failure()
        number = self.sequence
        self.sequence += 1
        # Two alternating mailboxes suffice. A rank cannot enter round n+2
        # until every rank entered n+1, and each rank enters n+1 only after it
        # has finished reading n. This bounds Store memory without an extra
        # acknowledgement/barrier or racing deletion against a slow reader.
        prefix = f"round/{number % 2}"
        try:
            message = json.dumps(
                {"round": number, "phase": phase, "value": value}, allow_nan=False
            )
            self.store.set(f"{prefix}/{self.rank}", message)
        except BaseException as caught:
            self.fail(phase, caught)
            raise
        keys = [f"{prefix}/{rank}" for rank in self.members]
        deadline = time.monotonic() + self.timeout
        messages = []
        while True:
            self.check_failure()
            available = [self.store.check([key]) for key in keys]
            messages = [
                json.loads(self.store.get(key)) if ready else None
                for key, ready in zip(keys, available, strict=True)
            ]
            missing = [
                rank
                for rank, item in zip(self.members, messages, strict=True)
                if item is None or item["round"] != number
            ]
            if not missing:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                error = TimeoutError(
                    f"phase {phase!r}, round {number}: missing ranks {missing}"
                )
                self.fail(phase, error)
                raise PreparationError(str(error)) from error
            time.sleep(min(0.002, remaining))
        self.check_failure()
        complete = cast(list[dict[str, Any]], messages)
        phases = [message["phase"] for message in complete]
        if any(label != phase for label in phases):
            mismatch = ValueError(
                f"preparation phase mismatch at round {number}: "
                f"{dict(zip(self.members, phases, strict=True))}"
            )
            self.fail(phase, mismatch)
            raise PreparationError(str(mismatch)) from mismatch
        return tuple(message["value"] for message in complete)

    def agree(self, phase: str, value: Any) -> Any:
        values = self.exchange(phase, value)
        if any(item != values[0] for item in values):
            mismatch = ValueError(
                f"participants disagree during {phase}: "
                f"{dict(zip(self.members, values, strict=True))}"
            )
            self.fail(phase, mismatch)
            raise PreparationError(str(mismatch)) from mismatch
        return values[0]

    def run(self, phase: str, action: Callable[[], T]) -> T:
        """Publish a local failure before peers wait for this phase's completion."""
        self.agree(phase + "/begin", phase)
        try:
            result = action()
        except BaseException as caught:
            self.fail(phase, caught)
            raise
        self.exchange(phase + "/ready", True)
        return result
