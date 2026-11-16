"""Pinned metadata, explicit overrides and CPU-only supplied model resolution."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from shadowspill.training.definitions import load_registry
from shadowspill_client import LoRA, Model, Weights
from shadowspill_workloads.definitions import pretrained
from shadowspill_workloads.initialization import hub

REVISION = "a" * 40
FAMILIES = ("llama3", "qwen3moe", "qwen35moe", "olmoe")


def metadata(family):
    result = {
        "model_type": {
            "llama3": "llama",
            "qwen3moe": "qwen3_moe",
            "qwen35moe": "qwen3_5_moe_text",
            "olmoe": "olmoe",
        }[family],
        "num_hidden_layers": 2,
        "hidden_size": 32,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 97,
        "intermediate_size": 16,
        "moe_intermediate_size": 16,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "rope_theta": 10_000,
        "max_position_embeddings": 128,
        "norm_topk_prob": family != "olmoe",
    }
    if family == "qwen35moe":
        result.update(
            shared_expert_intermediate_size=16,
            full_attention_interval=2,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            linear_conv_kernel_dim=4,
            layer_types=["linear_attention", "full_attention"],
        )
        return {"model_type": "qwen3_5_moe", "text_config": result}
    return result


@pytest.mark.parametrize("family", FAMILIES)
def test_public_resolution_pins_metadata_preserves_overrides_and_rechecks(
    monkeypatch, family
):
    calls = []

    def read(reference, revision):
        calls.append((reference, revision))
        return REVISION, metadata(family)

    monkeypatch.setattr(hub, "config", read)
    registry = load_registry(["shadowspill-workloads"])
    requested = replace(
        Model.from_pretrained(
            "fixture/model",
            revision="release",
            lora=LoRA(2),
            config={"architecture": {"max_seq_len": 32}},
            precision={"compute_dtype": "float32"},
            implementation="mlops",
        ),
        trainable=["blocks.*"],
    )
    resolved, definition, _ = registry.resolve_model_reference(requested)
    assert definition == "shadowspill/huggingface@1"
    assert resolved.definition == f"shadowspill/{family}@1"
    assert resolved.initialize == Weights.pretrained("fixture/model", revision=REVISION)
    assert resolved.config["architecture"]["max_seq_len"] == 32
    assert resolved.config["architecture"]["n_layers"] == 2
    assert resolved.config["architecture"]["d_model"] == 32
    assert (resolved.precision, resolved.lora, resolved.trainable) == (
        requested.precision,
        requested.lora,
        requested.trainable,
    )
    model = registry.resolve_model(resolved.definition, settings=resolved)
    repeated, _, _ = registry.resolve_model_reference(model.settings)
    assert repeated == model.settings
    assert calls == [("fixture/model", "release"), ("fixture/model", REVISION)]


@pytest.mark.parametrize("family", FAMILIES)
def test_metadata_sets_capacity_and_rejects_incompatible_checkpoint_overrides(family):
    actual, values = pretrained.architecture(metadata(family), {})
    assert actual == family
    assert values["max_seq_len"] == 128
    for field in ("n_layers", "d_model", "vocab_size"):
        with pytest.raises(ValueError, match=f"checkpoint geometry: {field}"):
            pretrained.architecture(metadata(family), {field: values[field] + 1})


@pytest.mark.parametrize(
    "field, value",
    [
        ("hidden_act", "gelu"),
        ("tie_word_embeddings", True),
        ("rope_scaling", {"factor": 8}),
        ("mlp_only_layers", [1]),
        ("decoder_sparse_step", 2),
        ("max_position_embeddings", -1),
        ("layer_types", ["full_attention", "linear_attention"]),
    ],
)
def test_unimplemented_checkpoint_semantics_are_rejected(field, value):
    config = metadata("qwen35moe")
    config["text_config"][field] = value
    with pytest.raises(ValueError):
        pretrained.architecture(config, {})


@pytest.mark.parametrize(
    "reference",
    [
        "/tmp/model",
        "../model",
        "file:///tmp/model",
        "https://example.com/model",
        "owner/../model",
        "owner/model/weights",
        "owner/model--copy",
    ],
)
def test_hub_predicate_rejects_paths_urls_and_noncanonical_ids(reference):
    assert not hub.accepts(reference)


def test_download_is_metadata_only_pinned_and_temporary(tmp_path, monkeypatch):
    import huggingface_hub as provider

    monkeypatch.setattr(provider, "try_to_load_from_cache", lambda *a, **k: None)
    monkeypatch.setattr(
        provider,
        "get_hf_file_metadata",
        lambda *a, **k: SimpleNamespace(
            commit_hash=REVISION,
            size=100,
        ),
    )
    downloads = []

    def download(reference, filename, *, revision, cache_dir):
        downloads.append((reference, filename, revision, cache_dir))
        path = Path(cache_dir) / filename
        path.write_text(json.dumps(metadata("llama3")))
        return str(path)

    monkeypatch.setattr(provider, "hf_hub_download", download)
    revision, values = hub.config("fixture/model", "release")
    assert revision == REVISION
    assert values["model_type"] == "llama"
    assert len(downloads) == 1
    assert downloads[0][:3] == ("fixture/model", "config.json", REVISION)
    assert not Path(downloads[0][3]).exists()


@pytest.mark.parametrize(
    "revision, size",
    [
        ("main", 100),
        ("b" * 40, 100),
        (REVISION, None),
        (REVISION, hub.CONFIG_BYTES + 1),
        (REVISION, -1),
    ],
)
def test_invalid_remote_size_or_revision_fails_before_download(
    monkeypatch, revision, size
):
    import huggingface_hub as provider

    monkeypatch.setattr(provider, "try_to_load_from_cache", lambda *a, **k: None)
    monkeypatch.setattr(
        provider,
        "get_hf_file_metadata",
        lambda *a, **k: SimpleNamespace(
            commit_hash=revision,
            size=size,
        ),
    )
    monkeypatch.setattr(
        provider, "hf_hub_download", lambda *a, **k: pytest.fail("download")
    )
    with pytest.raises(ValueError):
        hub.config("fixture/model", REVISION)


def test_pinned_cache_is_read_only_and_bounded(tmp_path, monkeypatch):
    import huggingface_hub as provider

    path = tmp_path / "config.json"
    path.write_text(json.dumps(metadata("llama3")))
    monkeypatch.setattr(provider, "try_to_load_from_cache", lambda *a, **k: str(path))
    monkeypatch.setattr(
        provider, "get_hf_file_metadata", lambda *a, **k: pytest.fail("network")
    )
    assert hub.config("fixture/model", REVISION) == (REVISION, metadata("llama3"))
    path.write_bytes(b" " * (hub.CONFIG_BYTES + 1))
    with pytest.raises(ValueError, match="metadata byte limit"):
        hub.config("fixture/model", REVISION)


def test_resolution_imports_no_tensor_framework():
    code = f"""
import builtins, sys
original = builtins.__import__
blocked = ('torch', 'triton', 'mlops', 'transformers')
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in blocked:
        raise RuntimeError('framework import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from shadowspill.training.definitions import load_registry
from shadowspill_client import Model
from shadowspill_workloads.initialization import hub
hub.config = lambda *a: ({REVISION!r}, {metadata("qwen35moe")!r})
registry = load_registry(['shadowspill-workloads'])
model, _, _ = registry.resolve_model_reference(Model.from_pretrained('fixture/model'))
resolved = registry.resolve_model(model.definition, settings=model)
assert resolved.config['architecture']['max_seq_len'] == 128
assert not any(name in sys.modules for name in blocked)
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=60)
