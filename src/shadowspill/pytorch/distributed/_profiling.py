"""Preparation-only coordination around real task invocations and caches."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from . import current

_PHASE: ContextVar[tuple[str, ...]] = ContextVar(
    "shadowspill_profile_phase", default=()
)


@contextmanager
def phase(name: str) -> Iterator[None]:
    token = _PHASE.set((*_PHASE.get(), name))
    try:
        yield
    finally:
        _PHASE.reset(token)


def label(suffix: str) -> str:
    return "/".join(("profile", *_PHASE.get(), suffix))


def all_ready(suffix: str, ready: bool) -> bool:
    bound = current()
    return ready if bound is None else all(bound.control.exchange(label(suffix), ready))


def any_needed(suffix: str, needed: bool) -> bool:
    return not all_ready(suffix, not needed)


def continue_timing(needed: bool, allowed: bool) -> bool:
    bound = current()
    if bound is None:
        return needed and allowed
    values = bound.control.exchange(label("continue"), [needed, allowed])
    return any(item[0] for item in values) and all(item[1] for item in values)


@contextmanager
def invocation() -> Iterator[None]:
    """The caller drains device work before leaving this scope."""
    bound = current()
    if bound is not None:
        bound.control.agree(label("invoke/begin"), None)
    try:
        yield
    except BaseException as error:
        if bound is not None:
            bound.control.fail(label("invoke"), error)
        raise
    if bound is not None:
        bound.control.exchange(label("invoke/complete"), True)


@dataclass(frozen=True)
class Case:
    position: int
    occurrences: tuple[int, ...]
    identity: str


def cases(
    artifacts: Sequence[object], local_keys: Sequence[str], *, purpose: str
) -> tuple[Case, ...]:
    """Deduplicate the complete participant vector in occurrence order."""
    bound = current()
    assert bound is not None
    records = bound.control.exchange(
        f"{purpose}/inventory",
        [
            [getattr(item, "kind", type(item).__name__), key]
            for item, key in zip(artifacts, local_keys, strict=True)
        ],
    )
    roles = [[item[0] for item in rank] for rank in records]
    if any(value != roles[0] for value in roles):
        raise ValueError(
            "distributed preparation requires matching ordered task roles on all ranks"
        )
    grouped: dict[str, list[int]] = {}
    for position in range(len(artifacts)):
        payload = [bound.control.members, [rank[position] for rank in records]]
        identity = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()
        grouped.setdefault(identity, []).append(position)
    return tuple(
        Case(positions[0], tuple(positions), identity)
        for identity, positions in grouped.items()
    )
