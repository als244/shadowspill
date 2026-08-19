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


def test_tooling_releases_the_case_model_without_a_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = nn.Linear(4, 4)
    case = _Case(model)
    calls: dict[str, object] = {}

    def release_state(
        value: nn.Module,
        *,
        runtime: object,
    ) -> None:
        calls.update(model=value, runtime=runtime)

    monkeypatch.setattr(model_state, "release_model_state", release_state)
    runtime = object()
    model_state.release_case_model(case, runtime=cast(Any, runtime))

    assert calls == {"model": model, "runtime": runtime}
