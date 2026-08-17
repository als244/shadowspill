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


def test_tooling_replaces_case_source_with_imported_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = nn.Linear(4, 4)
    imported = nn.Linear(4, 4)
    source_id = id(source)
    source_reference = weakref.ref(source)
    calls: dict[str, object] = {}

    def import_state(
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
        return imported

    monkeypatch.setattr(model_state, "import_model_state", import_state)
    runtime = object()
    case = _Case(source)
    case = model_state.import_case_model(
        case,
        runtime=cast(Any, runtime),
    )
    del source
    gc.collect()

    assert source_reference() is None
    assert case.model is imported
    assert calls == {
        "source_id": source_id,
        "runtime": runtime,
        "pool": "spill",
        "release_source": True,
    }


def test_tooling_exports_the_case_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = nn.Linear(4, 4)
    case = _Case(model)
    calls: dict[str, object] = {}

    def export_state(
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

    monkeypatch.setattr(model_state, "export_model_state", export_state)
    runtime = object()
    result = model_state.export_case_model(
        case,
        runtime=cast(Any, runtime),
    )

    assert result is model
    assert calls == {
        "model": model,
        "runtime": runtime,
        "release_runtime": True,
    }
