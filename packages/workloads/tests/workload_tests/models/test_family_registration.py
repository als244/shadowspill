"""Family metadata stays lightweight and pins affected training definitions."""

import subprocess
import sys
from dataclasses import replace

import pytest

from shadowspill.training.definitions import load_registry
from shadowspill_workloads.definitions import source
from shadowspill_workloads.models import families


def test_catalog_and_family_metadata_load_without_framework_or_engine():
    script = """
import builtins, sys, json
from dataclasses import asdict
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'triton', 'mlops', 'shadowspill'):
        raise AssertionError('metadata imported runtime code: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from shadowspill_workloads.models import catalog
from shadowspill_workloads.models.families import families, checkpoint_families
assert catalog.names() and checkpoint_families()
assert all(family.name for family in families())
for name in catalog.names():
    config = catalog.entry(name).config()
    restored = type(config)(**json.loads(json.dumps(asdict(config))))
    assert config == restored, name
assert not any('.implementations.' in name for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=60)


@pytest.mark.parametrize(
    "family",
    ["llama3", "qwen3", "qwen35", "olmoe", "qwen3moe", "qwen35moe", "regression"],
)
def test_family_registration_changes_invalidate_saved_model_source(monkeypatch, family):
    original = load_registry(["shadowspill-workloads"])
    resolved = original.resolve_model(family)
    other = original.resolve_model("qwen3" if family != "qwen3" else "llama3")
    add = source._add

    def changed(digest, name, path):
        add(digest, name, path)
        if name == f"models/{family}/registration.py":
            digest.update(b"changed family declaration")

    monkeypatch.setattr(source, "_add", changed)
    current = load_registry(["shadowspill-workloads"])
    with pytest.raises(ValueError, match="source changed"):
        current.resolve_model(family, source=resolved.source)
    assert current.resolve_model(other.definition.id).source == other.source
    _, old_source = original.model_reference("shadowspill/huggingface@1")
    with pytest.raises(ValueError, match="source changed"):
        current.model_reference("shadowspill/huggingface@1", source=old_source)


def test_duplicate_pretrained_identifiers_fail_instead_of_selecting_a_family(
    monkeypatch,
):
    first, second = families.families()[:2]
    monkeypatch.setattr(
        families,
        "families",
        lambda: (first, replace(second, checkpoint_types=first.checkpoint_types)),
    )
    with pytest.raises(ValueError, match="registered twice"):
        families.checkpoint_families()


@pytest.mark.parametrize(
    "metadata, family",
    [
        ({"model_type": "glm_moe_dsa"}, "glm53"),
        ({"model_type": "glm5_next_text"}, "glm53_flash"),
        (
            {
                "model_type": "glm5_next",
                "text_config": {"model_type": "glm5_next_text"},
            },
            "glm53_flash",
        ),
    ],
)
def test_glm_checkpoint_identification_uses_the_family_declaration(metadata, family):
    from shadowspill_workloads.models.catalog.sources import checkpoint_family

    assert checkpoint_family(metadata) == family
