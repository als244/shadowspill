"""Public pretrained initialization reuses bounded readers across supplied models."""

import hashlib
import importlib
import json
import struct
from dataclasses import replace

import pytest
import torch
from safetensors.torch import save_file

from shadowspill.artifacts.tensors import TensorChunk
from shadowspill.parallelism import Parallelism
from shadowspill.training.definitions import (
    InitializationContext,
    InitializationRuntime,
    load_registry,
)
from shadowspill_client import LoRA, Model, Weights
from shadowspill_workloads.initialization import headers, hub, hub_files
from shadowspill_workloads.initialization.text import Initialize
from workload_tests.test_checkpoints import huggingface_state
from workload_tests.test_llama_definition import settings as llama_settings
from workload_tests.test_moe_definitions import settings as moe_settings
from workload_tests.test_pretrained_metadata import FAMILIES, REVISION


@pytest.mark.parametrize("family", FAMILIES)
def test_exported_weights_derived_buffers_require_no_weight_source(tmp_path, family):
    from shadowspill.training.worker.initialization import (
        PreparedInitialization,
        open_initialization,
    )

    registry = load_registry(["shadowspill-workloads"])
    requested = (
        llama_settings(capacity=8) if family == "llama3" else moe_settings(family)
    )
    requested = replace(requested, initialize=Weights.artifact("model-export"))
    resolved = registry.resolve_model(requested.definition, settings=requested)
    context = InitializationContext(
        resolved.settings,
        resolved.settings.initialize,
        Parallelism(),
        0,
        temporary=("rotary.cosine", "rotary.sine"),
    )
    description = resolved.describe_initialization(context)
    assert description.disk_bytes == 0 and description.peak_bytes < 64 << 20
    initial = PreparedInitialization(context, description, resolved.source)
    runtime = InitializationRuntime(tmp_path, lambda _: pytest.fail("weight source"))
    model = resolved.create()
    generated = Initialize(family, model.config, seed=29)
    with open_initialization(initial, runtime) as loader:
        for name, tensor in model.named_buffers():
            if name not in context.temporary:
                continue
            reference = torch.empty_like(tensor)
            fill(generated, name, reference, 19)
            fill(loader, name, tensor, 17)
            torch.testing.assert_close(tensor, reference, rtol=0, atol=0)
        with pytest.raises(ValueError, match="undeclared model state"):
            loader(
                TensorChunk("embed.weight", (1,), torch.float32, 0, 1), torch.zeros(1)
            )
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError, match="derived rotary buffers"):
        resolved.describe_initialization(replace(context, temporary=("unknown",)))


def fill(loader, name, tensor, chunk):
    for offset in range(0, tensor.numel(), chunk):
        destination = tensor.detach().view(-1)[offset : offset + chunk]
        loader(
            TensorChunk(
                name, tuple(tensor.shape), tensor.dtype, offset, tensor.numel()
            ),
            destination,
        )


def checkpoint(directory, family, model):
    module = importlib.import_module(f"shadowspill_workloads.models.{family}.config")
    metadata = module.to_huggingface(model.config)
    (directory / "config.json").write_text(json.dumps(metadata))
    values = huggingface_state(family, model, model.config)
    shards = [dict(), dict()]
    index = {}
    for number, (name, value) in enumerate(values.items()):
        shard = number % 2
        shards[shard][name] = value
        index[name] = f"weights-{shard}.safetensors"
    for number, shard in enumerate(shards):
        save_file(shard, directory / f"weights-{number}.safetensors")
    (directory / hub_files.INDEX).write_text(json.dumps({"weight_map": index}))
    manifest = [
        {
            "name": path.name,
            "size": path.stat().st_size,
            "checksum": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(directory.iterdir())
    ]
    return metadata, manifest


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_pretrained_chunks_reproduce_weights_rotary_and_fresh_lora(
    tmp_path, monkeypatch, family, lora, dtype
):
    import huggingface_hub as provider

    registry = load_registry(["shadowspill-workloads"])
    desired = (
        llama_settings(capacity=8, lora=LoRA(2) if lora else None)
        if family == "llama3"
        else moe_settings(family, lora=lora)
    )
    base = registry.resolve_model(
        desired.definition, settings=replace(desired, lora=None)
    ).create()
    initialize = Initialize(family, base.config, seed=17)
    for name, tensor in (*base.named_parameters(), *base.named_buffers()):
        fill(initialize, name, tensor, 4093)
    source, cache = tmp_path / "source", tmp_path / "private"
    source.mkdir()
    cache.mkdir()
    metadata, manifest = checkpoint(source, family, base)
    monkeypatch.setattr(hub, "config", lambda *a: (REVISION, metadata))
    monkeypatch.setattr(hub_files, "describe", lambda *a: manifest)
    monkeypatch.setattr(
        provider, "try_to_load_from_cache", lambda ref, name, **k: str(source / name)
    )
    requested = Model.from_pretrained(
        f"fixture/{family}",
        lora=desired.lora,
        config={"architecture": {"max_seq_len": 8}},
        precision={"compute_dtype": dtype, "lora_dtype": "float32"},
    )
    pinned, _, _ = registry.resolve_model_reference(requested)
    resolved = registry.resolve_model(pinned.definition, settings=pinned)
    initial = resolved.describe_initialization(
        InitializationContext(
            resolved.settings,
            resolved.settings.initialize,
            Parallelism(),
            0,
        )
    )
    runtime = InitializationRuntime(
        cache, lambda _: pytest.fail("artifact"), initial.disk_bytes
    )
    assert initial.peak_bytes <= 256 << 20
    actual = resolved.create()
    expected = dict((*base.named_parameters(), *base.named_buffers()))
    generated = Initialize(family, actual.config, seed=0)
    with initial.open(runtime) as loader:
        for name, tensor in (*actual.named_parameters(), *actual.named_buffers()):
            reference = expected.get(name)
            if reference is None:
                reference = torch.empty_like(tensor)
                fill(generated, name, reference, max(1, tensor.numel()))
            fill(loader, name, tensor, 17)
            torch.testing.assert_close(
                tensor, reference.to(tensor.dtype), rtol=0, atol=0
            )
    assert not list(cache.iterdir())
    for file in hub_files.files(manifest):
        hub_files.verify(source / file.name, file)


@pytest.mark.parametrize("fault", ["length", "dimensions", "extra", "index"])
def test_header_preflight_rejects_before_the_tensor_reader(tmp_path, fault):
    (tmp_path / "config.json").write_text("{}")
    header = {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    if fault == "dimensions":
        header["weight"]["shape"] = [1] * 9
    if fault == "extra":
        header["weight"]["unexpected"] = {"nested": [1]}
    payload = json.dumps(header).encode()
    length = headers.HEADER_BYTES + 1 if fault == "length" else len(payload)
    shard = tmp_path / "model.safetensors"
    shard.write_bytes(struct.pack("<Q", length) + payload + bytes(4))
    manifest = [hub_files.HubFile("model.safetensors", shard.stat().st_size, "a" * 64)]
    if fault == "index":
        (tmp_path / hub_files.INDEX).write_text(
            json.dumps({"weight_map": {"weight": "other.safetensors"}})
        )
        manifest.append(hub_files.HubFile(hub_files.INDEX, 100, "b" * 64))
    with pytest.raises(
        ValueError, match=r"(byte limit|dimensions|declaration|shard manifest)"
    ):
        headers.validate(tmp_path, manifest)
