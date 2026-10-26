"""Portable descriptions, group validation and transactional provider lifetimes."""

import json
from dataclasses import replace

import pytest
import torch

from shadowspill import Parallelism, Program, Resource
from shadowspill.resources.factories import sources
from shadowspill.resources.inventory import resource_inventory
from shadowspill.resources.lifecycle import ResourceBindings
from tests.shadowspill.offline._example import program


def test_description_roundtrip_does_not_import_provider(tmp_path):
    parallel = Parallelism.dp(ranks_per_replica=2)
    resource = Resource(
        "not_installed.backend:open",
        {"capacity": 32},
        {"team": parallel.group("all_ranks")},
    )
    artifact = replace(program(), parallelism=parallel, resources={"lookup": resource})
    artifact.save(tmp_path / "program.ssp")
    restored = Program.load(tmp_path / "program.ssp")
    assert restored.resources == artifact.resources
    assert restored.digest == artifact.digest
    with pytest.raises(TypeError):
        restored.resources["new"] = resource
    with pytest.raises(TypeError):
        resource.options["capacity"] = 64
    with pytest.raises(ValueError, match="outside parallelism"):
        replace(program(), resources={"lookup": resource})
    with pytest.raises(TypeError, match="finite JSON"):
        Resource("not_installed:open", {"tensor": torch.ones(1)})


def test_resources_close_after_partial_initialization_failure(tmp_path):
    factory = "tests.shadowspill.resources._provider:initialize"
    options = {"events": str(tmp_path / "events"), "elements": 16}
    descriptions = {
        "a": Resource(factory, options),
        "b": Resource(factory, {**options, "fail": True}),
    }
    live = ResourceBindings()
    with pytest.raises(ValueError, match="intentional initialization failure"):
        live.open(descriptions, sources(descriptions), device=torch.device("cpu"))
    live.close()
    assert [
        json.loads(line)["event"]
        for line in (tmp_path / "events").read_text().splitlines()
    ] == ["open", "close"]


def test_factory_changes_reject_before_initialization(tmp_path):
    descriptions = {
        "lookup": Resource(
            "tests.shadowspill.resources._provider:initialize",
            {"events": str(tmp_path / "events"), "elements": 16},
        )
    }
    with pytest.raises(ValueError, match="factory source differs"):
        ResourceBindings().open(
            descriptions, {"lookup": "another-build"}, device=torch.device("cpu")
        )
    assert not (tmp_path / "events").exists()


def test_shared_resources_require_identical_descriptions_and_sources():
    first = replace(
        program(),
        resources={"lookup": Resource("provider:open", {"size": 2})},
        bindings={"resource_sources": {"lookup": "build1"}},
    )
    resources, identities = resource_inventory([first, first])
    assert resources == first.resources and identities == {"lookup": "build1"}
    with pytest.raises(ValueError, match="conflicting descriptions"):
        resource_inventory(
            [first, replace(first, bindings={"resource_sources": {"lookup": "build2"}})]
        )
