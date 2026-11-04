"""Backend settings separate compiler artifacts and offline profiling records."""

import pytest
import torch

from shadowspill.build.compilation import compile_task
from shadowspill.build.compilation.inductor import compiler
from shadowspill.build.lowering.catalog import CompiledTasks
from shadowspill.pytorch import numerics
from tests.shadowspill.build.compilation.test_compiler import Add, artifact


def test_reduction_setting_separates_aot_cache_and_profile_identity(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path))
    captured = artifact(Add(), (torch.randn(8, 8), torch.randn(8, 8)))
    first = compile_task(captured)
    matmul = torch.backends.cuda.matmul
    monkeypatch.setattr(
        matmul,
        "allow_bf16_reduced_precision_reduction",
        not matmul.allow_bf16_reduced_precision_reduction,
    )
    second = compile_task(captured)

    def unexpected(*args, **kwargs):
        raise AssertionError("warm numeric configuration rebuilt GraphLowering")

    monkeypatch.setattr(compiler, "_graph_lowering_contract", unexpected)
    warm = compile_task(captured)
    assert second.manifest == warm.manifest
    assert second.numerical_contracts == warm.numerical_contracts
    assert first.manifest.compatibility_digest != second.manifest.compatibility_digest
    catalogs = [
        CompiledTasks({captured.compatibility_digest: built}, {})
        for built in (first, second, warm)
    ]
    keys = [catalog.key(captured) for catalog in catalogs]
    assert keys[0] != keys[1] == keys[2]
    for catalog, key, built in zip(catalogs, keys, (first, second, warm), strict=True):
        assert catalog.bindings()["task_metadata"][key]["numerical_contracts"] == (
            built.numerical_contracts
        )


@pytest.mark.parametrize("name", list(numerics._settings()))
def test_every_recorded_setting_changes_compiled_and_profile_keys(monkeypatch, name):
    captured = artifact(Add(), (torch.randn(2, 3), torch.randn(2, 3)))
    original = compile_task(captured)
    values = numerics._settings()
    old = values[name]
    values[name] = (
        not old
        if isinstance(old, bool)
        else old + 1
        if isinstance(old, int)
        else "different"
    )
    monkeypatch.setattr(numerics, "_settings", lambda: values)
    changed = compile_task(captured)
    keys = [
        CompiledTasks({captured.compatibility_digest: item}, {}).key(captured)
        for item in (original, changed)
    ]
    assert (
        original.manifest.compatibility_digest != changed.manifest.compatibility_digest
    )
    assert keys[0] != keys[1]
