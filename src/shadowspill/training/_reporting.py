"""Bounded, asynchronous CPU observations; independent of task coordination."""

from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from typing import Any

import torch.distributed as dist
from torch._C._distributed_c10d import PrefixStore
from torch.distributed.distributed_c10d import _get_process_group_store

Record = dict[str, Any]
Reducer = Callable[[Sequence[Mapping[str, Any]]], Mapping[str, Any]]


def sum_rank_records(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Add objective/work contributions; report the slowest local step duration.

    Every train/loss must already use the caller's global normalizer. Other
    metrics remain rank-local unless an application supplies its own reducer.
    Work units are optional and application-defined, never inferred as tokens.
    """
    result: Record = {"step": records[0]["step"]}
    for name in ("train/loss", "train/work_units"):
        if all(name in record for record in records):
            result[name] = sum(record[name] for record in records)
    for name in ("train/step_seconds", "train/elapsed_seconds", "eval/seconds"):
        if all(name in record for record in records):
            result[name] = max(record[name] for record in records)
    if result.get("train/step_seconds", 0) > 0 and "train/work_units" in result:
        result["train/units_per_second"] = (
            result["train/work_units"] / result["train/step_seconds"]
        )
    return result


class RecordExchange:
    """Publish CPU records without waiting for other ranks in the caller.

    A dedicated store namespace and bounded mailboxes avoid interfering with
    preparation/checkpoint sequencing. Threads never inspect device tensors or
    issue device operations. Matching records are combined on the first rank.
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        emit: Callable[[Record], None],
        reducer: Reducer,
        *,
        max_pending: int,
        timeout: float,
    ) -> None:
        if str(dist.get_backend(group)).lower() != "gloo":
            raise ValueError("distributed reporting needs a CPU Gloo group")
        if max_pending < 1 or timeout <= 0:
            raise ValueError("reporting capacity and timeout must be positive")
        self.members = tuple(dist.get_process_group_ranks(group))
        self.rank = dist.get_rank()
        if self.rank not in self.members:
            raise ValueError("reporting rank is not a member of the group")
        identity = [uuid.uuid4().hex if self.rank == self.members[0] else None]
        # Initialization is coordinated. The training loop has no reporting
        # collective/barrier: only background threads touch the store below.
        dist.broadcast_object_list(identity, src=self.members[0], group=group)
        self.store = PrefixStore(
            f"shadowspill/reporting/{identity[0]}/", _get_process_group_store(group)
        )
        self.store.set(f"ack/{self.rank}", "-1")
        self.max_pending, self.timeout = max_pending, timeout
        self.emit, self.reducer = emit, reducer
        self.sequence = 0
        self.pending: queue.Queue[str] = queue.Queue(maxsize=max_pending)
        self.stopped = threading.Event()
        self.error: Exception | None = None
        self.closed = False
        self.sender = threading.Thread(target=self._guard_send, daemon=True)
        self.collector = (
            threading.Thread(target=self._guard_collect, daemon=True)
            if self.rank == self.members[0]
            else None
        )
        self.sender.start()
        if self.collector is not None:
            self.collector.start()

    def submit(self, record: Record) -> None:
        if self.closed:
            raise RuntimeError("distributed logger is closed")
        self._raise_error()
        # Serialization rejects device tensors without reading/synchronizing them.
        message = json.dumps({"sequence": self.sequence, "record": record})
        try:
            self.pending.put_nowait(message)
        except queue.Full as error:
            raise RuntimeError(
                "distributed reporting queue is full; reduce logging frequency "
                "or increase max_pending"
            ) from error
        self.sequence += 1

    def _raise_error(self) -> None:
        if self.error is not None:
            raise RuntimeError(
                f"distributed reporting failed: {self.error}"
            ) from self.error

    def _check(self) -> None:
        if self.stopped.is_set():
            raise RuntimeError("distributed reporting stopped")
        if self.store.check(["failure"]):
            raise RuntimeError(self.store.get("failure").decode())

    def _failed(self, error: Exception) -> None:
        self.error = error
        # Keep the original failure observable even if the store itself died.
        with suppress(Exception):
            self.store.set("failure", f"rank {self.rank}: {error}")

    def _guard_send(self) -> None:
        try:
            self._send()
        except Exception as error:
            self._failed(error)

    def _send(self) -> None:
        while not self.stopped.is_set():
            self._check()
            try:
                message = self.pending.get(timeout=0.02)
            except queue.Empty:
                continue
            envelope = json.loads(message)
            sequence = envelope["sequence"]
            deadline = time.monotonic() + self.timeout
            while sequence - int(self.store.get(f"ack/{self.rank}")) > self.max_pending:
                self._check()
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "aggregate reporting did not consume its mailbox"
                    )
                self.stopped.wait(0.005)
            self.store.set(f"record/{self.rank}/{sequence % self.max_pending}", message)
            if "done" in envelope:
                while int(self.store.get(f"ack/{self.rank}")) < sequence:
                    self._check()
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "aggregate reporting did not finish flushing"
                        )
                    self.stopped.wait(0.005)
                return

    def _guard_collect(self) -> None:
        try:
            self._collect()
        except Exception as error:
            self._failed(error)

    def _collect(self) -> None:
        sequence = 0
        waiting_since: float | None = None
        while not self.stopped.is_set():
            self._check()
            messages = []
            for rank in self.members:
                key = f"record/{rank}/{sequence % self.max_pending}"
                if self.store.check([key]):
                    message = json.loads(self.store.get(key))
                    if message["sequence"] == sequence:
                        messages.append(message)
            if messages and waiting_since is None:
                waiting_since = time.monotonic()
            if len(messages) != len(self.members):
                if (
                    waiting_since is not None
                    and time.monotonic() - waiting_since > self.timeout
                ):
                    raise TimeoutError(
                        f"missing rank observations at record {sequence}"
                    )
                self.stopped.wait(0.005)
                continue
            finished = ["done" in message for message in messages]
            if any(finished):
                if not all(finished):
                    raise ValueError(
                        "ranks emitted different numbers of reporting records"
                    )
                errors = [message["done"] for message in messages if message["done"]]
                if errors:
                    raise RuntimeError(
                        f"a reporting rank ended unsuccessfully: {errors}"
                    )
            else:
                records = [message["record"] for message in messages]
                steps = [record["step"] for record in records]
                phases = [
                    "train"
                    if "train/loss" in r
                    else "eval"
                    if "eval/loss" in r
                    else "other"
                    for r in records
                ]
                if len(set(steps)) != 1 or len(set(phases)) != 1:
                    raise ValueError("rank reporting step/phase order differs")
                combined = dict(self.reducer(records))
                if combined.get("step") != steps[0]:
                    raise ValueError("aggregate reducer must retain the completed step")
                if len(combined) > 1:
                    self.emit(combined)
            for rank in self.members:
                self.store.set(f"ack/{rank}", str(sequence))
            if all(finished):
                return
            sequence += 1
            waiting_since = None

    def close(self, *, exit_code: int = 0) -> None:
        if self.closed:
            return
        self.closed = True
        deadline = time.monotonic() + self.timeout
        try:
            self._raise_error()
            self.pending.put(
                json.dumps({"sequence": self.sequence, "done": exit_code}),
                timeout=self.timeout,
            )
            for thread in (self.sender, self.collector):
                if thread is not None:
                    thread.join(timeout=max(0.0, deadline - time.monotonic()))
                    if thread.is_alive():
                        raise TimeoutError("distributed reporting shutdown timed out")
            self._raise_error()
        finally:
            self.stopped.set()
