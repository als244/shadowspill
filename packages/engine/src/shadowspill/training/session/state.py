"""Operation boundaries; no transport, device dispatch or implicit optimizer."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Literal
from uuid import uuid4

from shadowspill_client.protocol.requests import identifier
from shadowspill_client.protocol.results import Accumulation

type Command = Literal[
    "step",
    "forward_backward",
    "forward",
    "optim_step",
    "discard",
    "save",
    "restore",
    "close",
]
_COMMANDS = frozenset(
    (
        "step",
        "forward_backward",
        "forward",
        "optim_step",
        "discard",
        "save",
        "restore",
        "close",
    )
)


@dataclass(frozen=True)
class Boundary:
    history_id: str
    completed_updates: int = 0
    accumulation: Accumulation = field(default_factory=Accumulation)
    valid: bool = True
    closed: bool = False


@dataclass(frozen=True)
class Transition:
    """Decision at the queue head, valid only for its issuing state machine.

    A skipped compute can still require a prepared output-only invocation. It
    must not dispatch backward/optimizer tasks. Completion follows that work
    and required terminal transfers; publication happens after this boundary.
    """

    request_id: str
    command: Command
    before: Boundary
    participates: bool
    accumulate_gradients: bool
    skipped: bool


class OperationState:
    """One in-flight command, with mutation committed at confirmed completion.

    The coordinator owns validation, deduplication, durable request order and
    dependency failures. This component owns live accumulation and update
    counters. A worker failure invalidates them; a validation rejection does
    not. A restored checkpoint can rewind updates, never generation identities.
    """

    def __init__(
        self, *, completed_updates: int = 0, history_id: str | None = None
    ) -> None:
        self._updates(completed_updates)
        self._boundary = Boundary(
            identifier(
                "history-" + uuid4().hex if history_id is None else history_id,
                "history ID",
            ),
            completed_updates=completed_updates,
        )
        self._pending: Transition | None = None

    @property
    def boundary(self) -> Boundary:
        return self._boundary

    def begin(
        self, command: Command, request_id: str, *, participates: bool | None = None
    ) -> Transition:
        identifier(request_id, "request ID")
        if command not in _COMMANDS:
            raise ValueError(f"unknown session command: {command}")
        if self._pending is not None:
            raise RuntimeError("another session command is in flight")
        before = self._boundary
        if before.closed:
            raise RuntimeError("session is closed")
        if not before.valid and command not in ("restore", "close"):
            raise RuntimeError("session state is invalid; restore or start a new run")
        accumulation = before.accumulation
        if (
            before.valid
            and accumulation.open
            and command in ("step", "save", "restore", "close")
        ):
            raise RuntimeError(
                "finish optimization or explicitly discard gradients first"
            )
        if command in ("step", "forward_backward"):
            if type(participates) is not bool:
                raise TypeError("compute requires validated global participation")
        elif participates is not None:
            raise ValueError("only step/backward commands accept participation")
        if command == "optim_step":
            if not accumulation.open:
                raise RuntimeError("optim_step requires an open accumulation")
            participates = accumulation.participating_contributions > 0
        if (
            command == "forward_backward"
            and accumulation.open
            and request_id in accumulation.contributions
        ):
            raise ValueError("contribution request ID already completed; do not replay")
        pending = Transition(
            request_id,
            command,
            before,
            bool(participates),
            command == "forward_backward"
            and bool(participates)
            and accumulation.participating_contributions > 0,
            command in ("step", "forward_backward", "optim_step") and not participates,
        )
        self._pending = pending
        return pending

    def complete(
        self, pending: Transition, *, restored_updates: int | None = None
    ) -> Boundary:
        self._boundary = self.completion(pending, restored_updates=restored_updates)
        self._pending = None
        return self._boundary

    def completion(
        self, pending: Transition, *, restored_updates: int | None = None
    ) -> Boundary:
        """Project this queue-head completion for admission without mutating state."""
        self._require_pending(pending)
        command, before = pending.command, pending.before
        if command == "restore":
            self._updates(restored_updates)
        elif restored_updates is not None:
            raise ValueError("only restore sets completed checkpoint updates")
        accumulation, updates = before.accumulation, before.completed_updates
        if command == "forward_backward":
            accumulation = Accumulation(
                generation=accumulation.generation + (not accumulation.open),
                open=True,
                contributions=(
                    *(accumulation.contributions if accumulation.open else ()),
                    pending.request_id,
                ),
                participating_contributions=(
                    accumulation.participating_contributions if accumulation.open else 0
                )
                + pending.participates,
            )
        elif command in ("optim_step", "discard", "restore"):
            accumulation = Accumulation(generation=accumulation.generation)
        if command in ("step", "optim_step"):
            updates += pending.participates
        elif command == "restore":
            assert restored_updates is not None
            updates = restored_updates
        # Stable across repeated admission projections, unique to this restore.
        # Publication does not own or change the authoritative history boundary.
        history = before.history_id
        if command == "restore":
            history = (
                "history-"
                + hashlib.sha256(
                    (history + "\0" + pending.request_id).encode()
                ).hexdigest()
            )
        return Boundary(
            history,
            updates,
            accumulation,
            before.valid or command == "restore",
            command == "close",
        )

    def fail(self, pending: Transition, *, execution_started: bool) -> Boundary:
        """Unknown partial effects invalidate state; rejection preserves it.

        Call only before completion. A later artifact failure reports the
        already committed boundary and must never roll it back or rerun work.
        """
        self._require_pending(pending)
        if type(execution_started) is not bool:
            raise TypeError("execution_started must be boolean")
        if execution_started:
            self._boundary = replace(self._boundary, valid=False)
        self._pending = None
        return self._boundary

    def invalidate(self) -> None:
        """Stop after an uncertain receipt or lost authoritative outcome commit."""
        self._boundary = replace(self._boundary, valid=False)
        self._pending = None

    def _require_pending(self, pending: Transition) -> None:
        if self._pending is not pending:
            raise RuntimeError("transition is not the current session command")

    @staticmethod
    def _updates(value: int | None) -> None:
        if type(value) is not int or value < 0:
            raise ValueError("completed updates must be a nonnegative integer")
