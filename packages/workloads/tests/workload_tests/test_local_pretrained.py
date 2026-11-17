"""Local references pin content, preserve sources and use bounded shared readers."""

import hashlib
import json
import os
import struct
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from shadowspill.training.definitions import InitializationRuntime, load_registry
from shadowspill_client import LoRA, Model
from shadowspill_workloads.initialization import local_files, weight_files
from workload_tests.test_pretrained_metadata import FAMILIES, metadata


def checkpoint(root, family="qwen3moe"):
    (root / "config.json").write_text(json.dumps(metadata(family)))
    header = json.dumps(
        {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    ).encode()
    (root / "model.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header + bytes(4)
    )


@pytest.mark.parametrize("family", FAMILIES)
def test_local_resolution_pins_content_and_rechecks_metadata_without_payload_scan(
    tmp_path, monkeypatch, family
):
    checkpoint(tmp_path, family)
    registry = load_registry(["shadowspill-workloads"])
    requested = Model.from_pretrained(
        str(tmp_path), lora=LoRA(2), config={"architecture": {"max_seq_len": 32}}
    )
    pinned, definition, _ = registry.resolve_model_reference(requested)
    assert definition == "shadowspill/local_safetensors@1"
    assert pinned.definition == f"shadowspill/{family}@1"
    assert pinned.lora == requested.lora
    assert pinned.config["architecture"]["max_seq_len"] == 32
    manifest = weight_files.files(pinned.initialize.metadata["files"])
    assert pinned.initialize.revision == local_files.revision(manifest)
    for file in manifest:
        assert (
            file.checksum
            == hashlib.sha256((tmp_path / file.name).read_bytes()).hexdigest()
        )
    monkeypatch.setattr(
        local_files, "_checksum", lambda *a: pytest.fail("payload rescan")
    )
    resolved = registry.resolve_model(pinned.definition, settings=pinned)
    assert registry.resolve_model_reference(resolved.settings)[0] == resolved.settings
    with pytest.raises(ValueError, match="content revision"):
        registry.resolve_model_reference(
            replace(pinned, initialize=replace(pinned.initialize, revision="changed"))
        )
    changed = metadata(family)
    changed.get("text_config", changed)["max_position_embeddings"] += 1
    (tmp_path / "config.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match=r"pinned manifest|differs"):
        registry.resolve_model_reference(pinned)


def test_worker_verifies_payload_before_loader_and_leaves_source_unchanged(tmp_path):
    source, cache = tmp_path / "source", tmp_path / "cache"
    source.mkdir()
    cache.mkdir()
    checkpoint(source)
    reference, _, manifest = local_files.describe(str(source))
    # A reader without an index must not discover unrelated checkpoint shards.
    (source / "unrelated.safetensors").write_bytes(b"unrelated")
    runtime = InitializationRuntime(
        cache, lambda _: pytest.fail("artifact"), local_files.disk_bytes(manifest)
    )
    expected = {file.name: (source / file.name).read_bytes() for file in manifest}
    with local_files.snapshot(reference, manifest, runtime) as private:
        assert {p.name for p in private.iterdir()} == expected.keys()
        assert all((private / name).is_symlink() for name in expected)
        assert {name: (private / name).read_bytes() for name in expected} == expected
    assert not list(cache.iterdir())
    assert {name: (source / name).read_bytes() for name in expected} == expected
    shard = source / "model.safetensors"
    facts = shard.stat()
    shard.write_bytes(shard.read_bytes()[:-1] + b"\x01")
    os.utime(shard, ns=(facts.st_atime_ns, facts.st_mtime_ns))
    # Setup retains its content pin; worker admission verifies payloads even if
    # the bytes changed without changing the filename, size or mtime.
    assert local_files.describe(reference, [asdict(f) for f in manifest])[2] == manifest
    with (
        pytest.raises(ValueError, match="checksum differs"),
        local_files.snapshot(reference, manifest, runtime),
    ):
        pytest.fail("loader opened corrupted weights")
    assert not list(cache.iterdir())


def test_local_metadata_bounds_precede_payload_reads(tmp_path, monkeypatch):
    checkpoint(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", 5 << 20) + b"bad")
    monkeypatch.setattr(
        local_files, "_checksum", lambda *a: pytest.fail("payload scan")
    )
    with pytest.raises(ValueError, match="metadata byte limit"):
        local_files.describe(str(tmp_path))


def test_first_content_fingerprint_streams_at_most_one_mib(tmp_path, monkeypatch):
    path = tmp_path / "model.safetensors"
    payload = bytes(range(256)) * (9 << 10)
    path.write_bytes(payload)
    original = Path.open
    reads = []

    class Guard:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def fileno(self):
            return self.stream.fileno()

        def seek(self, offset):
            return self.stream.seek(offset)

        def read(self, size=-1):
            assert 0 <= size <= 1 << 20
            reads.append(size)
            return self.stream.read(size)

    def guarded(source, *args, **kwargs):
        stream = original(source, *args, **kwargs)
        return Guard(stream) if source == path else stream

    monkeypatch.setattr(Path, "open", guarded)
    declared = weight_files.WeightFile(path.name, len(payload), "0" * 64)
    result = local_files._checksum(tmp_path, declared)
    assert result.checksum == hashlib.sha256(payload).hexdigest()
    assert len(reads) >= 4 and max(reads) == 1 << 20
    with pytest.raises(ValueError, match="SHA256"):
        local_files.revision([replace(result, checksum="a" * 40)])


def test_changed_source_during_initialization_is_rejected_and_private_links_removed(
    tmp_path,
):
    source, cache = tmp_path / "source", tmp_path / "cache"
    source.mkdir()
    cache.mkdir()
    checkpoint(source)
    reference, _, manifest = local_files.describe(str(source))
    runtime = InitializationRuntime(
        cache, lambda _: pytest.fail("artifact"), local_files.disk_bytes(manifest)
    )
    with (
        pytest.raises(ValueError, match="changed during initialization"),
        local_files.snapshot(reference, manifest, runtime),
    ):
        path = source / "model.safetensors"
        path.write_bytes(path.read_bytes()[:-1] + b"\x01")
    assert not list(cache.iterdir())
    with (
        pytest.raises(ValueError, match="admitted initialization disk"),
        local_files.snapshot(reference, manifest, replace(runtime, disk_bytes=0)),
    ):
        pytest.fail("unadmitted files")


@pytest.mark.parametrize(
    "reference",
    ["relative/model", "~/model", "file:///tmp/model", "https://example.com/model"],
)
def test_local_predicate_requires_an_absolute_server_path(reference):
    assert not local_files.accepts(reference)


def test_local_resolution_needs_no_framework_or_network(tmp_path):
    checkpoint(tmp_path)
    code = """
import builtins, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'triton', 'mlops', 'transformers'):
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from shadowspill.training.definitions import load_registry
from shadowspill_client import Model
from shadowspill_workloads.initialization import hub
hub.config = lambda *a: (_ for _ in ()).throw(AssertionError('network'))
registry = load_registry(['shadowspill-workloads'])
model, definition, source = registry.resolve_model_reference(
    Model.from_pretrained(sys.argv[1]))
assert definition == 'shadowspill/local_safetensors@1'
resolved = registry.resolve_model(model.definition, settings=model)
assert resolved.config['architecture']['d_model'] == 32
"""
    subprocess.run([sys.executable, "-c", code, str(tmp_path)], check=True, timeout=60)
