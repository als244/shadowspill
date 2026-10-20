"""The catalog names every supplied model, and every name resolves."""

from __future__ import annotations

from importlib import import_module

import pytest
import torch

from workloads.full_model import spec
from workloads.models import catalog


def test_names_are_family_then_size_and_unique() -> None:
    assert len(set(catalog.names())) == len(catalog.names())
    for item in catalog.entries():
        prefix = item.family.replace("_", "-")
        expected = prefix if item.size == "published" else f"{prefix}-{item.size}"
        assert item.name == expected, (item.name, expected)


@pytest.mark.parametrize("item", catalog.entries(), ids=lambda item: item.name)
def test_every_entry_resolves_to_its_preset_and_implementations(item) -> None:
    config_module = import_module(f"workloads.models.{item.family}.config")
    assert isinstance(item.config(), config_module.Config)
    assert item.size in config_module.PRESETS
    for implementation in item.implementations:
        assert item.model_class(implementation).__name__ == "Model"
    missing = set(catalog.IMPLEMENTATIONS) - set(item.implementations)
    for implementation in missing:
        with pytest.raises(ValueError, match="has no"):
            item.model_class(implementation)


def test_the_gate_cells_keep_their_families_and_identities() -> None:
    for family in ("llama3", "qwen35", "olmoe"):
        assert f"{family}-numerical" in catalog.names()
    assert spec("llama3-8b", "mlops").identity == "mlops_llama3"
    assert spec("qwen35-9b", "pytorch").identity == "pytorch_qwen35"
    assert spec("olmoe-7b-a1b", "mlops").family == "olmoe"
    with pytest.raises(ValueError, match="has no 'pytorch'"):
        spec("qwen3moe-30b-a3b", "pytorch")
    with pytest.raises(ValueError, match="unknown model"):
        spec("llama3-8B", "mlops")


def test_configuration_changes_reach_the_entry_and_bad_fields_are_refused() -> None:
    manifest = spec("llama3-tiny", "pytorch", n_layers=1, max_seq_len=64)
    assert manifest.model_config.n_layers == 1
    assert manifest.model_config.max_seq_len == 64
    with pytest.raises(ValueError, match="llama3-tiny"):
        spec("llama3-tiny", "pytorch", layers=1)


@pytest.mark.parametrize(
    "item",
    [
        item
        for item in catalog.entries()
        if item.size == "tiny" and item.factory is None
    ],
    ids=lambda item: item.name,
)
def test_tiny_shapes_construct_on_meta_in_every_implementation(item) -> None:
    for implementation in item.implementations:
        model = catalog.build_on_meta(item.name, implementation, dtype="bfloat16")
        parameters = list(model.parameters())
        assert parameters and all(p.is_meta for p in parameters)
        # A mixer may keep its own state in FP32; the weights follow the dtype.
        assert model.embed.weight.dtype == torch.bfloat16


def test_build_on_meta_applies_lora_and_configuration_changes() -> None:
    model = catalog.build_on_meta(
        "llama3-tiny",
        "mlops",
        dtype="bfloat16",
        lora={"rank": 4, "alpha": 4, "factor_dtype": "float32"},
        max_seq_len=64,
    )
    assert model.config.max_seq_len == 64
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora" in name for name in trainable)


def test_mixtures_and_expert_parallel_follow_the_families() -> None:
    for item in catalog.entries():
        assert item.mixture == (item.family not in {"llama3", "qwen35"})
        assert item.expert_parallel == (
            item.family in {"olmoe", "qwen3moe", "qwen35moe"}
        )
    assert set(catalog.expert_parallel_names()) == {
        item.name for item in catalog.entries() if item.expert_parallel
    }


def test_factory_entries_name_an_importable_factory() -> None:
    for item in catalog.entries():
        if item.factory is None:
            continue
        module, _, name = item.factory.partition(":")
        assert callable(getattr(import_module(module), name))
        assert item.implementations == ("mlops",)
