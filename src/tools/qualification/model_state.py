"""Runtime-state ownership helpers for qualification model cases."""

from __future__ import annotations

import copy
from dataclasses import is_dataclass, replace
from typing import Any, cast

import torch.nn as nn

from shadowspill.pytorch import (
    Runtime,
    import_model_state,
    release_model_state,
)


def import_case_model[CaseT](
    case: CaseT,
    *,
    runtime: Runtime,
    pool: str = "spill",
) -> CaseT:
    """Return ``case`` with its source model replaced by runtime-owned state.

    The returned case must replace the caller's original case reference. This
    prevents a case container from accidentally retaining the anonymous CPU
    model after ``release_source=True`` import.
    """

    source = _case_model(case)
    imported = import_model_state(
        source,
        runtime=runtime,
        pool=pool,
        release_source=True,
    )
    try:
        updated = _replace_case_model(case, imported)
    except BaseException:
        release_model_state(imported, runtime=runtime)
        raise
    return cast(CaseT, updated)


def release_case_model(case: object, *, runtime: Runtime) -> None:
    """Release the case's runtime-owned model without an anonymous CPU copy.

    Qualification evidence is captured through ``state_dict()`` copies before
    teardown, so the model itself is never reused after its protocol. An
    export copy cannot coexist with the full pinned spill arena on
    qualification hosts. The case must be discarded after this call.
    """

    release_model_state(_case_model(case), runtime=runtime)


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
        raise RuntimeError("qualification case did not adopt the imported model")
    return updated


def _case_model(case: object) -> nn.Module:
    model: Any = getattr(case, "model", None)
    if not isinstance(model, nn.Module):
        raise TypeError("qualification case must expose an nn.Module as .model")
    return model


__all__ = ["import_case_model", "release_case_model"]
