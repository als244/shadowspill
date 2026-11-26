"""Checksum/byte limits and cleanup apply to real HTTP streams and local caches."""

import hashlib
import importlib
import json
from contextlib import contextmanager
from dataclasses import asdict
from functools import partial

import pytest

from shadowspill.training.components import InitializationRuntime
from shadowspill_workloads.initialization import hub_files, weight_files
from workload_tests.initialization._hub_server import serve_hub
from workload_tests.initialization.test_pretrained_metadata import REVISION


def declaration(name, data, *, git=False):
    digest = (
        hashlib.sha1(f"blob {len(data)}\0".encode() + data)
        if git
        else hashlib.sha256(data)
    )
    return weight_files.WeightFile(name, len(data), digest.hexdigest())


@contextmanager
def http_stream(monkeypatch, chunks, *, encoding="identity"):
    from huggingface_hub import utils

    http = importlib.import_module(type(utils.get_session()).__module__.split(".")[0])

    class Stream(http.SyncByteStream):
        def __iter__(self):
            yield from chunks

    def respond(request):
        assert request.method == "GET" and REVISION in request.url.path
        assert request.headers["Accept-Encoding"] == "identity"
        return http.Response(
            200, headers={"Content-Encoding": encoding}, stream=Stream()
        )

    with http.Client(transport=http.MockTransport(respond)) as client:
        monkeypatch.setattr(utils, "get_session", lambda: client)
        yield


@pytest.mark.parametrize("git", [False, True])
def test_bounded_http_download_and_cached_content_hash(tmp_path, monkeypatch, git):
    data = b"content for a pinned model" * 100000
    file = declaration("model.safetensors", data, git=git)
    path = tmp_path / file.name
    with http_stream(monkeypatch, (data[:12345], data[12345:])):
        hub_files.download("fixture/model", REVISION, file, path)
    assert path.read_bytes() == data
    weight_files.verify(path, file)
    path.write_bytes(b"changed" + data[7:])
    with pytest.raises(ValueError, match="checksum"):
        weight_files.verify(path, file)


@pytest.mark.parametrize("mode", ["oversize", "truncated", "corrupt", "encoded"])
def test_download_rejects_invalid_data_without_writing_beyond_the_limit(
    tmp_path, monkeypatch, mode
):
    data = b"expected bytes"
    file = declaration("model.safetensors", data)
    received = {
        "oversize": data + b"too many",
        "truncated": data[:-1],
        "corrupt": b"?" * len(data),
        "encoded": data,
    }[mode]
    path = tmp_path / file.name
    with (
        http_stream(
            monkeypatch,
            (received,),
            encoding="gzip" if mode == "encoded" else "identity",
        ),
        pytest.raises(ValueError, match=r"(size|checksum|encoding)"),
    ):
        hub_files.download("fixture/model", REVISION, file, path)
    assert path.stat().st_size <= file.size


@pytest.mark.parametrize("fail", [False, True])
def test_snapshot_reuses_verified_cache_and_cleans_owned_files(
    tmp_path, monkeypatch, fail
):
    import huggingface_hub as provider

    source = tmp_path / "existing-cache"
    source.mkdir()
    config, data = b'{"model_type":"fixture"}', b"checkpoint bytes"
    (source / "config.json").write_bytes(config)
    (source / "model.safetensors").write_bytes(data)
    manifest = (
        declaration("config.json", config),
        declaration("model.safetensors", data),
    )
    private = tmp_path / "private"
    private.mkdir()
    runtime = InitializationRuntime(
        private, lambda _: pytest.fail("artifact"), weight_files.disk_bytes(manifest)
    )
    monkeypatch.setattr(
        provider, "try_to_load_from_cache", lambda ref, name, **k: str(source / name)
    )
    monkeypatch.setattr(
        hub_files, "download", lambda *a: pytest.fail("downloaded cached file")
    )
    if fail:
        (source / "model.safetensors").write_bytes(b"incorrect bytes!")
        with (
            pytest.raises(ValueError, match=r"(size|checksum)"),
            hub_files.snapshot("fixture/model", REVISION, manifest, runtime),
        ):
            pytest.fail("opened invalid cached data")
    else:
        with hub_files.snapshot(
            "fixture/model", REVISION, manifest, runtime
        ) as snapshot:
            assert (snapshot / "model.safetensors").is_symlink()
            assert (snapshot / "model.safetensors").read_bytes() == data
    assert not list(private.iterdir())
    assert (source / "config.json").read_bytes() == config
    assert (source / "model.safetensors").exists()


@pytest.mark.parametrize(
    "name",
    [
        "../weight.safetensors",
        "/weight.safetensors",
        "a//w.safetensors",
        "a/./w.safetensors",
        "a\\w.safetensors",
        "weights.bin",
    ],
)
def test_manifests_reject_paths_outside_the_supported_format(name):
    with pytest.raises(ValueError, match="file name"):
        weight_files.WeightFile(name, 10, "a" * 64)


def test_index_and_manifest_bounds_reject_before_payload_access(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="metadata byte limit"):
        weight_files.WeightFile(
            weight_files.INDEX, weight_files.INDEX_BYTES + 1, "a" * 64
        )
    path = tmp_path / weight_files.INDEX
    path.write_text(json.dumps({"weight_map": {"weight": "../escape.safetensors"}}))
    with pytest.raises(ValueError, match="invalid tensor or shard"):
        weight_files.read_index(path)
    metadata = declaration("config.json", b"{}")
    with pytest.raises(ValueError, match="unique names"):
        weight_files.files([asdict(metadata), asdict(metadata)])
    monkeypatch.setattr(weight_files, "MAX_SHARDS", 1)
    manifest = [
        asdict(declaration(name, b"{}"))
        for name in ("config.json", weight_files.INDEX, "a.safetensors")
    ]
    for optional in ([], [asdict(declaration(weight_files.QUANT_CONFIG, b"{}"))]):
        assert len(weight_files.files(manifest + optional)) == 3 + len(optional)
    with pytest.raises(ValueError, match="shard count"):
        weight_files.files([*manifest, asdict(declaration("b.safetensors", b"{}"))])
    path.write_text(
        json.dumps({"weight_map": {"a": "a.safetensors", "b": "b.safetensors"}})
    )
    with pytest.raises(ValueError, match="shard count"):
        weight_files.read_index(path)


@pytest.mark.parametrize("sharded", [False, True])
def test_real_http_metadata_describes_without_fetching_weight_payloads(
    tmp_path, monkeypatch, sharded
):
    import huggingface_hub as provider

    from shadowspill_workloads.initialization import hub

    source, cache = tmp_path / "source", tmp_path / "private"
    source.mkdir()
    cache.mkdir()
    (source / "config.json").write_text('{"model_type":"llama"}')
    name = "part.safetensors" if sharded else "model.safetensors"
    (source / name).write_bytes(b"weight payload")
    if sharded:
        (source / weight_files.INDEX).write_text(
            json.dumps({"weight_map": {"weight": name}})
        )
    monkeypatch.setattr(provider, "try_to_load_from_cache", lambda *a, **k: None)
    with serve_hub(source) as (endpoint, requests):
        monkeypatch.setattr(
            provider, "hf_hub_url", partial(provider.hf_hub_url, endpoint=endpoint)
        )
        assert hub.config("fixture/llama", REVISION)[1] == {"model_type": "llama"}
        manifest = weight_files.files(hub_files.describe("fixture/llama", REVISION))
        assert ("HEAD", name) in requests and ("GET", name) not in requests
        runtime = InitializationRuntime(
            cache, lambda _: pytest.fail("artifact"), weight_files.disk_bytes(manifest)
        )
        with hub_files.snapshot(
            "fixture/llama", REVISION, manifest, runtime
        ) as snapshot:
            assert (snapshot / name).read_bytes() == b"weight payload"
        assert ("GET", name) in requests
    assert not list(cache.iterdir())
