"""Bind Quack's measured configs without consulting its mutable tuning caches.

This adapter covers the checked Autotuner call contract, including generated
GEMM epilogues. It records plain metadata, not provider objects or kernel code.
Unknown callable state and candidates generated only by pruning need explicit
provider support; they must not silently fall back to tuning during replay.
"""

import hashlib
import importlib
import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields, is_dataclass
from enum import Enum
from functools import lru_cache, partial, wraps
from threading import Lock
from types import CodeType, FunctionType, MethodType
from typing import Any

import torch

from shadowspill.artifacts._values import digest

from ._scope import active

_CALL_SOURCE = "f35d8c4030c2d27ddd57f74e0026368229400f0ddf936ce3fe8d670bf313dbc8"
_lock = Lock()
_installed = False


def _name(value: Any) -> str:
    return f"{value.__module__}.{value.__qualname__}"


@lru_cache(maxsize=256)
def _code(code: CodeType) -> str:
    # File paths and line numbers differ between build and worker checkouts.
    # Include nested code and the execution-relevant fields, never repr(code).
    return digest(
        {
            "bytecode": code.co_code.hex(),
            "constants": _value(code.co_consts),
            "names": code.co_names,
            "variables": code.co_varnames,
            "free": code.co_freevars,
            "cells": code.co_cellvars,
            "arguments": (
                code.co_argcount,
                code.co_posonlyargcount,
                code.co_kwonlyargcount,
            ),
            "flags": code.co_flags,
            "exceptions": code.co_exceptiontable.hex(),
        }
    )


def _value(value: Any) -> Any:
    """Stable, JSON-only identity for provider metadata; never tensor contents."""
    if isinstance(value, Enum):
        return {"enum": _name(type(value)), "value": _value(value.value)}
    if value is None or isinstance(value, str | bool | int | float):
        return value
    if value is Ellipsis:
        return {"ellipsis": True}
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, torch.dtype):
        return {"dtype": str(value)}
    if isinstance(value, tuple | list):
        return {type(value).__name__: [_value(item) for item in value]}
    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        return {key: _value(item) for key, item in value.items()}
    if isinstance(value, CodeType):
        return {"code": _code(value)}
    if isinstance(value, FunctionType):
        return {
            "function": _name(value),
            "name": value.__name__,
            "code": _code(value.__code__),
            "defaults": _value(value.__defaults__),
            "kwdefaults": _value(value.__kwdefaults__),
            "closure": {
                name: _value(cell.cell_contents)
                for name, cell in zip(
                    value.__code__.co_freevars, value.__closure__ or (), strict=True
                )
            },
        }
    if isinstance(value, MethodType):
        return {"method": _value(value.__func__), "self": _value(value.__self__)}
    if isinstance(value, partial):
        return {
            "partial": _value(value.func),
            "args": _value(value.args),
            "kwargs": _value(value.keywords),
        }
    semantic = getattr(value, "semantic_digest", None)
    if isinstance(semantic, str) and semantic:
        return {"type": _name(type(value)), "semantic_digest": semantic}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "dataclass": _name(type(value)),
            "fields": {
                field.name: _value(getattr(value, field.name))
                for field in fields(value)
            },
        }
    # Generated epilogues close over sink allocation methods. Quack gives
    # EpiOps a checked static configuration identity, including sink options.
    operations = importlib.import_module("quack.epilogue.ops")
    if isinstance(value, operations.EpiOp):
        return {"epilogue_op": _value(value.cache_key())}
    raise ValueError(
        f"Quack tuning metadata {_name(type(value))} needs a provider initializer"
    )


@lru_cache(maxsize=8)
def _source_for_dirs(directories: tuple[str, ...]) -> str:
    # Recompute when registered kernel-source roots change. Quack's own cache is
    # process-wide and may have been populated before a provider registered them.
    module = importlib.import_module("quack.cache.jit")
    return str(module._compute_source_fingerprint.__wrapped__())


def _provider_source() -> str:
    module = importlib.import_module("quack.cache")
    return _source_for_dirs(tuple(str(path) for path in module.EXTRA_SOURCE_DIRS))


def _identity(tuner: Any, args: Sequence[Any], kwargs: Mapping[str, Any]) -> str:
    # Match the checked provider's key exactly: scalar keys are keyword-only;
    # tensor geometry follows positional order, then named keyword order.
    key = [kwargs[name] for name in tuner.keys if name in kwargs]
    tensors = (
        *args,
        *(value for name, value in kwargs.items() if name in tuner._arg_name_set),
    )
    for value in tensors:
        if isinstance(value, torch.Tensor):
            key.extend(
                (
                    tuple(value.shape),
                    tuple(min(stride, 2) for stride in value.stride()),
                    value.dtype,
                )
            )
    return "quack/v1/" + digest(
        {
            "function": _value(tuner.fn),
            "arguments": tuner.arg_names,
            "keys": tuner.keys,
            "key": _value(tuple(key)),
        }
    )


def _run(
    original: Callable[..., Any],
    supported: bool,
    tuner: Any,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
) -> Any:
    scope = active.get()
    if scope is None:
        return original(tuner, *args, **kwargs)
    if not supported:
        raise ValueError(
            "unsupported Quack Autotuner call contract; update the provider adapter"
        )
    mode, choices = scope
    identity = _identity(tuner, args, kwargs)
    name = f"{tuner.fn.__module__}.{tuner.fn.__name__}"
    source = _provider_source()
    if mode == "capture":
        result = original(tuner, *args, **kwargs)
        selected = _value(tuner.best_config.all_kwargs())
        indices = [
            index
            for index, config in enumerate(tuner.configs)
            if _value(config.all_kwargs()) == selected
        ]
        if not indices:
            raise ValueError(
                f"{name}: generated Quack candidate needs a provider initializer"
            )
        choice = {
            "backend": "quack",
            "version": 1,
            "kernel": name,
            "source": source,
            "candidate": indices[0],
            "config": selected,
        }
        if identity in choices and choices[identity] != choice:
            raise ValueError(
                f"{name}: conflicting Quack configurations within one task"
            )
        choices[identity] = choice
        return result
    if identity not in choices:
        raise ValueError(
            f"{name}: no finalized configuration for this task input; "
            "rebuild the Program"
        )
    choice = choices[identity]
    if choice.get("backend") != "quack" or choice.get("version") != 1:
        raise ValueError(f"{name}: unsupported saved Quack choice version")
    if choice["source"] != source:
        raise ValueError(
            f"{name}: Quack provider source differs from the saved Program"
        )
    index = choice["candidate"]
    if (
        type(index) is not int
        or not 0 <= index < len(tuner.configs)
        or _value(tuner.configs[index].all_kwargs()) != choice["config"]
    ):
        raise ValueError(f"{name}: Quack candidates differ from the saved Program")
    config = tuner.configs[index]
    # This is the checked provider's ordinary invocation, after selection.
    # Do not modify cache entries, prune, benchmark, or consult a disk cache.
    tuner.best_config = config
    result = tuner.fn(*args, **kwargs, **config.all_kwargs())
    tuner.nargs = None
    return result


def install() -> None:
    global _installed
    with _lock:
        if _installed:
            return
        try:
            module = importlib.import_module("quack.autotuner")
        except ModuleNotFoundError as error:
            if error.name not in {"quack", "quack.autotuner"}:
                raise
            return
        original = module.Autotuner.__call__
        try:
            supported = (
                not hasattr(original, "__wrapped__")
                and hashlib.sha256(inspect.getsource(original).encode()).hexdigest()
                == _CALL_SOURCE
            )
        except (OSError, TypeError):
            supported = False

        @wraps(original)
        def call(tuner: Any, *args: Any, **kwargs: Any) -> Any:
            return _run(original, supported, tuner, args, kwargs)

        module.Autotuner.__call__ = call
        _installed = True
