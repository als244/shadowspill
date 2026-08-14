"""Runtime-state ownership helpers for qualification model cases."""

from __future__ import annotations

import copy
from dataclasses import is_dataclass, replace
from typing import Any, cast

import torch.nn as nn

from shadowspill.pytorch import (
    Runtime,
    externalize_model_state,
    relocate_model_state,
)


def relocate_case_model[CaseT](
    case: CaseT,
    *,
    runtime: Runtime,
    pool: str = "spill",
) -> CaseT:
    """Return ``case`` with its source model replaced by runtime-owned state.

    The returned case must replace the caller's original case reference. This
    prevents a case container from accidentally retaining the anonymous CPU
    model after ``release_source=True`` relocation.
    """

    source = _case_model(case)
    relocated = relocate_model_state(
        source,
        runtime=runtime,
        pool=pool,
        release_source=True,
    )
    try:
        updated = _replace_case_model(case, relocated)
    except BaseException:
        externalize_model_state(relocated, runtime=runtime, release_runtime=True)
        raise
    return cast(CaseT, updated)


def externalize_case_model(
    case: object,
    *,
    runtime: Runtime,
    release_runtime: bool = True,
) -> nn.Module:
    """Externalize the model currently owned by a qualification case."""

    return externalize_model_state(
        _case_model(case),
        runtime=runtime,
        release_runtime=release_runtime,
    )


def _replace_case_model(case: object, model: nn.Module) -> object:
    updated: Any
    if is_dataclass(case) and not isinstance(case, type):
        updated = replace(case, model=model)
    else:
        updated = copy.copy(case)
        try:
            updated.model = model
        except (AttributeError, TypeError) as exc:
            raise TypeError(
                "qualification case must be a dataclass or expose a writable "
                "model attribute"
            ) from exc
    if _case_model(updated) is not model:
        raise RuntimeError("qualification case did not adopt the relocated model")
    return updated


def _case_model(case: object) -> nn.Module:
    model: Any = getattr(case, "model", None)
    if not isinstance(model, nn.Module):
        raise TypeError("qualification case must expose an nn.Module as .model")
    return model


__all__ = ["externalize_case_model", "relocate_case_model"]
