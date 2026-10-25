"""Capture and bind autotuner winners; external kernel code stays in its provider.

The ContextVar scopes choices to one task, including when two saved plans choose
different configurations for the same kernel. Candidate identity and kernel
source are checked before replay. A missing choice fails before any benchmarking.
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from threading import Lock
from typing import Any

from triton.runtime.autotuner import Autotuner  # type: ignore[import-untyped]
from triton.runtime.jit import JITFunction  # type: ignore[import-untyped]

from shadowspill.artifacts._values import digest, plain

_active: ContextVar[tuple[str, dict[str, Any]] | None] = ContextVar(
    "shadowspill_triton_choices", default=None
)
_lock = Lock()
_installed = False


def _identity(
    tuner: Any, args: Sequence[Any], kwargs: Mapping[str, Any]
) -> tuple[str, tuple[Any, ...], str]:
    function = tuner.fn
    while not isinstance(function, JITFunction):
        function = function.fn
    arguments = {**dict(zip(tuner.arg_names, args, strict=False)), **kwargs}
    tracked = {key: value for key, value in arguments.items() if key in tuner.arg_names}
    key = tuple(tracked[name] for name in tuner.keys if name in tracked) + tuple(
        str(value.dtype) for value in tracked.values() if hasattr(value, "dtype")
    )
    identity = digest({"kernel": function.cache_key, "key": key})
    return identity, key, f"{function.__module__}.{function.__name__}"


def _config(config: Any) -> dict[str, Any]:
    # Hooks remain in the provider's candidate object, never serialized as code.
    values = {**vars(config), "pre_hook": config.pre_hook is not None}
    return {key: plain(value) for key, value in values.items()}


def _run(
    original: Callable[..., Any],
    tuner: Any,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
) -> Any:
    active = _active.get()
    if active is None or len(tuner.configs) <= 1:
        return original(tuner, *args, **kwargs)
    mode, choices = active
    identity, key, name = _identity(tuner, args, kwargs)
    if mode == "capture":
        result = original(tuner, *args, **kwargs)
        selected = _config(tuner.best_config)
        indices = [
            i for i, value in enumerate(tuner.configs) if _config(value) == selected
        ]
        if not indices:
            raise ValueError(
                f"{name}: external tuning config needs a provider initializer"
            )
        choices[identity] = {
            "kernel": name,
            "candidate": indices[0],
            "config": selected,
        }
        return result
    if identity not in choices:
        raise ValueError(f"{name}: no finalized configuration for this task input")
    choice = choices[identity]
    index = choice["candidate"]
    if index >= len(tuner.configs) or _config(tuner.configs[index]) != choice["config"]:
        raise ValueError(f"{name}: Triton candidates differ from the saved Program")
    previous = tuner.cache.get(key)
    tuner.cache[key] = tuner.configs[index]
    try:
        return original(tuner, *args, **kwargs)
    finally:
        if previous is None:
            tuner.cache.pop(key, None)
        else:
            tuner.cache[key] = previous


def _install() -> None:
    global _installed
    with _lock:
        if not _installed:
            original = Autotuner.run

            @wraps(original)
            def run(tuner: Any, *args: Any, **kwargs: Any) -> Any:
                return _run(original, tuner, args, kwargs)

            Autotuner.run = run
            _installed = True


@contextmanager
def capture_choices() -> Iterator[dict[str, Any]]:
    _install()
    choices: dict[str, Any] = {}
    token = _active.set(("capture", choices))
    try:
        yield choices
    finally:
        _active.reset(token)


def use_choices(
    function: Callable[..., Any], choices: Mapping[str, Any]
) -> Callable[..., Any]:
    _install()
    values = plain(choices)

    @wraps(function)
    def call(*args: Any, **kwargs: Any) -> Any:
        token = _active.set(("replay", values))
        try:
            return function(*args, **kwargs)
        finally:
            _active.reset(token)

    return call
