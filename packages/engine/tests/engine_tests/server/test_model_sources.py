"""Deployment checkpoint directories remain separate from mutable run storage."""

from dataclasses import replace
from importlib.metadata import PackageNotFoundError

import pytest

from engine_tests.training.test_setup_resolution import arguments
from shadowspill.server import Server, service
from shadowspill.server.startup.model_sources import local_model_source
from shadowspill_client import Model
from shadowspill_client.protocol.setup import TrainingSetup


def settings(reference="Qwen/Qwen3.5-9B", **options):
    return TrainingSetup(
        **{**arguments(), "model": Model.from_pretrained(reference, **options)}
    )


@pytest.mark.parametrize("namespace", [False, True])
def test_existing_checkpoints_are_selected_without_modification(tmp_path, namespace):
    directory = tmp_path / ("Qwen/Qwen3.5-9B" if namespace else "Qwen3.5-9B")
    directory.mkdir(parents=True)
    marker = directory / "config.json"
    marker.write_text('{"model_type":"test"}')
    requested = settings()
    resolved = local_model_source(requested, tmp_path)
    assert resolved.model.initialize.reference == str(directory)
    assert (
        replace(
            resolved.model.initialize, reference=requested.model.initialize.reference
        )
        == requested.model.initialize
    )
    assert marker.read_text() == '{"model_type":"test"}'
    assert requested.model.initialize.reference == "Qwen/Qwen3.5-9B"


def test_missing_or_explicitly_pinned_sources_use_the_installed_resolver(tmp_path):
    requested = settings()
    assert local_model_source(requested, tmp_path) is requested
    (tmp_path / "Qwen3.5-9B").mkdir()
    pinned = settings(revision="a" * 40)
    assert local_model_source(pinned, tmp_path) is pinned
    absolute = settings(str(tmp_path / "Qwen3.5-9B"))
    assert local_model_source(absolute, tmp_path) is absolute
    assert local_model_source(requested, None) is requested


def test_ambiguous_and_nondirectory_matches_fail_before_resolution(tmp_path):
    (tmp_path / "Qwen3.5-9B").touch()
    with pytest.raises(ValueError, match="not a directory"):
        local_model_source(settings(), tmp_path)
    (tmp_path / "Qwen/Qwen3.5-9B").mkdir(parents=True)
    with pytest.raises(ValueError, match="ambiguous"):
        local_model_source(settings(), tmp_path)


@pytest.mark.parametrize("catalog", [True, False])
def test_server_defaults_do_not_require_the_optional_catalog(
    tmp_path, monkeypatch, catalog
):
    monkeypatch.setenv("SHADOWSPILL_TOKEN", "secret" * 8)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")

    def installed(name):
        assert name == "shadowspill-workloads"
        if not catalog:
            raise PackageNotFoundError(name)
        return object()

    monkeypatch.setattr(service, "distribution", installed)
    server = Server(service_root=tmp_path / "runs", model_root=tmp_path)
    assert server.packages == (
        ("shadowspill", "shadowspill-workloads") if catalog else ("shadowspill",)
    )
    assert server.allocation.environment["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert server.model_root == tmp_path and not server.root.exists()
    custom = Server(service_root=tmp_path / "custom", workload_packages=["my-models"])
    assert custom.packages == ("my-models",)
