from __future__ import annotations

import gc
import weakref
from dataclasses import dataclass
from typing import Any, cast

import pytest
import torch.nn as nn

import tools.qualification.model_state as model_state


@dataclass(frozen=True, slots=True)
class _Case:
    model: nn.Module


def test_tooling_replaces_case_source_with_relocated_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = nn.Linear(4, 4)
    relocated = nn.Linear(4, 4)
    source_id = id(source)
    source_reference = weakref.ref(source)
    calls: dict[str, object] = {}

    def relocate(
        model: nn.Module,
        *,
        runtime: object,
        pool: str,
        release_source: bool,
    ) -> nn.Module:
        calls.update(
            source_id=id(model),
            runtime=runtime,
            pool=pool,
            release_source=release_source,
        )
        return relocated

    monkeypatch.setattr(model_state, "relocate_model_state", relocate)
    runtime = object()
    case = _Case(source)
    case = model_state.relocate_case_model(
        case,
        runtime=cast(Any, runtime),
    )
    del source
    gc.collect()

    assert source_reference() is None
    assert case.model is relocated
    assert calls == {
        "source_id": source_id,
        "runtime": runtime,
        "pool": "spill",
        "release_source": True,
    }


def test_tooling_externalizes_the_case_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = nn.Linear(4, 4)
    case = _Case(model)
    calls: dict[str, object] = {}

    def externalize(
        value: nn.Module,
        *,
        runtime: object,
        release_runtime: bool,
    ) -> nn.Module:
        calls.update(
            model=value,
            runtime=runtime,
            release_runtime=release_runtime,
        )
        return value

    monkeypatch.setattr(model_state, "externalize_model_state", externalize)
    runtime = object()
    result = model_state.externalize_case_model(
        case,
        runtime=cast(Any, runtime),
    )

    assert result is model
    assert calls == {
        "model": model,
        "runtime": runtime,
        "release_runtime": True,
    }
