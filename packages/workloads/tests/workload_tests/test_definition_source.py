"""Provider provenance includes loaded extension files without importing them."""

from types import SimpleNamespace

import pytest

from shadowspill.training.definitions import load_registry
from shadowspill_workloads.definitions import source


def test_provider_python_and_binary_changes_invalidate_fresh_source_identity(
    tmp_path, monkeypatch
):
    root = tmp_path / "provider"
    root.mkdir()
    python = root / "__init__.py"
    python.write_text("raise AssertionError('must not import provider')")
    binary = root / "provider.so"
    binary.write_bytes(b"first binary")
    monkeypatch.setattr(
        source,
        "find_spec",
        lambda _: SimpleNamespace(
            submodule_search_locations=[str(root)], origin=str(python)
        ),
    )
    try:
        source._provider_digest.cache_clear()
        first = source._provider_digest("test-provider")
        binary.write_bytes(b"changed binary")
        source._provider_digest.cache_clear()  # Fresh worker/controller installation.
        second = source._provider_digest("test-provider")
        assert first != second
        python.write_text("raise AssertionError('changed Python')")
        source._provider_digest.cache_clear()
        assert source._provider_digest("test-provider") != second
    finally:
        source._provider_digest.cache_clear()


def test_absent_provider_and_single_extension_have_distinct_sources(
    tmp_path, monkeypatch
):
    binary = tmp_path / "provider.so"
    binary.write_bytes(b"binary")
    try:
        monkeypatch.setattr(source, "find_spec", lambda _: None)
        source._provider_digest.cache_clear()
        absent = source._provider_digest("test-provider")
        monkeypatch.setattr(
            source,
            "find_spec",
            lambda _: SimpleNamespace(
                submodule_search_locations=None, origin=str(binary)
            ),
        )
        source._provider_digest.cache_clear()
        assert source._provider_digest("test-provider") != absent
    finally:
        source._provider_digest.cache_clear()


def test_shared_training_options_changes_invalidate_every_using_model(monkeypatch):
    names = ("llama3", "qwen3moe", "qwen35moe", "olmoe", "regression")
    original = load_registry(["shadowspill-workloads"])
    saved = {
        name: original.resolve_model(f"shadowspill/{name}@1").source for name in names
    }
    add = source._add

    def changed(digest, name, path):
        add(digest, name, path)
        if name == "definitions/training_options.py":
            digest.update(b"changed shared implementation")

    monkeypatch.setattr(source, "_add", changed)
    current = load_registry(["shadowspill-workloads"])
    for name in names:
        definition = f"shadowspill/{name}@1"
        if name == "regression":
            assert current.resolve_model(definition).source == saved[name]
        else:
            assert current.resolve_model(definition).source != saved[name]
            with pytest.raises(ValueError, match="model definition source changed"):
                current.resolve_model(definition, source=saved[name])
