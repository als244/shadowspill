"""Task-local choices for the PyTorch frontend's Triton compiler."""

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any

from shadowspill.artifacts._values import plain

active: ContextVar[tuple[str, dict[str, Any]] | None] = ContextVar(
    "shadowspill_triton_choices", default=None
)
pinned: ContextVar[dict[str, Any] | None] = ContextVar(
    "shadowspill_pinned_triton_choices", default=None
)


def _install() -> None:
    from . import triton

    triton.install()


@contextmanager
def capture_choices() -> Iterator[dict[str, Any]]:
    _install()
    choices: dict[str, Any] = {}
    token = active.set(("capture", choices))
    try:
        yield choices
    finally:
        active.reset(token)


@contextmanager
def pin_choices(choices: Mapping[str, Any]) -> Iterator[None]:
    """Require recorded choices, including inside nested build captures.

    Capture still records only the kernels it actually calls. Missing choices
    fail before tuning; leaving the scope restores ordinary selection. Saved
    task replay remains authoritative and must agree with an enclosing pin.
    """
    _install()
    values = _validated(choices)
    token = pinned.set(values)
    try:
        yield
    finally:
        pinned.reset(token)


def _validated(choices: Mapping[str, Any]) -> dict[str, Any]:
    values = plain(choices)
    if not isinstance(values, dict):
        raise ValueError("compiler tuning choices must be a mapping")
    for value in values.values():
        if not isinstance(value, dict) or set(value) != {
            "kernel",
            "candidate",
            "config",
        }:
            raise ValueError(
                "unsupported compiler tuning record; rebuild the Program with "
                "the external provider's operator_preparation adapter"
            )
    return values


def use_choices(
    function: Callable[..., Any], choices: Mapping[str, Any]
) -> Callable[..., Any]:
    _install()
    values = _validated(choices)

    # External providers own their manifests through OperatorPreparation. Refuse
    # historical mixed-format records instead of silently dropping their policy.
    @wraps(function)
    def call(*args: Any, **kwargs: Any) -> Any:
        token = active.set(("replay", values))
        try:
            return function(*args, **kwargs)
        finally:
            active.reset(token)

    return call
