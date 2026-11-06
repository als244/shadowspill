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


def use_choices(
    function: Callable[..., Any], choices: Mapping[str, Any]
) -> Callable[..., Any]:
    _install()
    values = plain(choices)
    # External providers own their manifests through OperatorPreparation. Refuse
    # historical mixed-format records instead of silently dropping their policy.
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

    @wraps(function)
    def call(*args: Any, **kwargs: Any) -> Any:
        token = active.set(("replay", values))
        try:
            return function(*args, **kwargs)
        finally:
            active.reset(token)

    return call
