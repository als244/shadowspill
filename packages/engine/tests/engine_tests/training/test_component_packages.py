"""Only configured installed distributions can register server components."""

import os
import subprocess
import sys
from importlib.metadata import EntryPoint
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine_tests.fixtures import external_component
from shadowspill.training.components import load_registry
from shadowspill.training.components import packages as loader


def test_registration_uses_only_the_explicit_distribution_allowlist(monkeypatch):
    calls = []
    other = EntryPoint(name="ignored", value="bad:hook", group="another.group")
    declared = [
        EntryPoint(name=n, value=f"trusted:{n}", group=loader.ENTRY_POINT_GROUP)
        for n in ("z", "a")
    ]
    monkeypatch.setattr(
        loader,
        "distribution",
        lambda name: SimpleNamespace(entry_points=[other, *declared]),
    )
    monkeypatch.setattr(
        EntryPoint, "load", lambda ep: lambda registry: calls.append(ep.name)
    )
    load_registry(["My_Project"])
    assert calls == ["a", "z"]
    calls.clear()
    assert load_registry([]) is not None
    assert not calls
    for names in (
        ["My_Project", "my-project"],
        ["python.module:factory"],
        ["/path/to/package"],
    ):
        with pytest.raises(ValueError):
            load_registry(names)
    with pytest.raises(TypeError):
        load_registry("my-project")


def test_invalid_or_missing_packages_do_not_partially_register(monkeypatch):
    calls = []

    def lookup(name):
        if name == "missing":
            raise loader.PackageNotFoundError(name)
        entry = EntryPoint(
            name="register", value="trusted:register", group=loader.ENTRY_POINT_GROUP
        )
        return SimpleNamespace(entry_points=[entry])

    monkeypatch.setattr(loader, "distribution", lookup)
    monkeypatch.setattr(
        EntryPoint, "load", lambda ep: lambda registry: calls.append(ep.name)
    )
    with pytest.raises(ValueError, match="not installed"):
        load_registry(["valid", "missing"])
    assert not calls
    monkeypatch.setattr(
        loader, "distribution", lambda _: SimpleNamespace(entry_points=[])
    )
    with pytest.raises(ValueError, match="must publish"):
        load_registry(["empty"])


def test_registration_errors_propagate_without_skipping_the_package(monkeypatch):
    entry = EntryPoint(
        name="register", value="trusted:register", group=loader.ENTRY_POINT_GROUP
    )
    monkeypatch.setattr(
        loader, "distribution", lambda _: SimpleNamespace(entry_points=[entry])
    )
    monkeypatch.setattr(EntryPoint, "load", lambda _: 3)
    with pytest.raises(TypeError, match="registration function"):
        load_registry(["invalid"])
    monkeypatch.setattr(EntryPoint, "load", lambda _: lambda registry: registry)
    with pytest.raises(TypeError, match="in place"):
        load_registry(["invalid"])

    def failed(registry):
        raise RuntimeError("installed component failure")

    monkeypatch.setattr(EntryPoint, "load", lambda _: failed)
    with pytest.raises(RuntimeError, match="installed component failure"):
        load_registry(["invalid"])


def test_external_installed_entry_point_loads_in_fresh_framework_free_process(tmp_path):
    # A temporary installed distribution provides actual importlib metadata and
    # source; no monkeypatch supplies the registration hook in the child.
    (tmp_path / "external_fixture.py").write_text(
        Path(external_component.__file__).read_text()
    )
    metadata = tmp_path / "external_classifier_fixture-1.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: external-classifier-fixture\nVersion: 1\n"
    )
    (metadata / "entry_points.txt").write_text(
        "[shadowspill.components]\nclassifier = external_fixture:register\n"
    )
    script = """
import sys
for name in ('torch', 'triton', 'mlops', 'shadowspill_workloads'):
    sys.modules[name] = None
from shadowspill.training.components import load_registry
registry = load_registry(['external-classifier-fixture'])
model = registry.resolve_model('external/image_classifier@1')
objective = registry.resolve_objective(
    'external/classification@1', model=model
)
assert model.inputs.fields[0].path == ('pixels',)
assert objective.component.padding_safe
print('Installed external entry point resolved without frameworks or catalog')
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
