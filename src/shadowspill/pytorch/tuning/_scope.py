"""Task-local provider choices shared by the autotuner adapters."""

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any

from shadowspill.artifacts._values import plain

active: ContextVar[tuple[str, dict[str, Any]] | None] = ContextVar(
    "shadowspill_provider_choices", default=None
)


def _install() -> None:
    from . import quack, triton

    triton.install()
    quack.install()


@contextmanager
def capture_choices() -> Iterator[dict[str, Any]]:
    _install()
    choices: dict[str, Any] = {}
    token = active.set(("capture", choices))
    try:
        yield choices
    finally:
        active.reset(token)


def use_choices(
    function: Callable[..., Any], choices: Mapping[str, Any]
) -> Callable[..., Any]:
    _install()
    values = plain(choices)

    @wraps(function)
    def call(*args: Any, **kwargs: Any) -> Any:
        token = active.set(("replay", values))
        try:
            return function(*args, **kwargs)
        finally:
            active.reset(token)

    return call
