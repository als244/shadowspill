"""Pinned weight-file manifests and bounded, private Hub downloads."""

import hashlib
import json
import re
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from shadowspill.checkpoint.inspection import check_json

CONFIG_BYTES = 1 << 20
INDEX_BYTES = 4 << 20
MAX_SHARDS = 512
MAX_TENSORS = 65536
CHUNK_BYTES = 1 << 20
INDEX = "model.safetensors.index.json"


def shard_name(value):
    return (
        isinstance(value, str)
        and 0 < len(value) <= 512
        and not value.startswith("/")
        and "\\" not in value
        and all(part not in ("", ".", "..") for part in value.split("/"))
        and value.endswith(".safetensors")
    )


@dataclass(frozen=True)
class HubFile:
    name: str
    size: int
    checksum: str

    def __post_init__(self):
        if self.name not in ("config.json", INDEX) and not shard_name(self.name):
            raise ValueError("invalid pretrained file name")
        if type(self.size) is not int or self.size <= 0:
            raise ValueError("pretrained file size must be positive")
        limit = {"config.json": CONFIG_BYTES, INDEX: INDEX_BYTES}.get(self.name)
        if limit is not None and self.size > limit:
            raise ValueError("pretrained JSON exceeds its metadata byte limit")
        if (
            not isinstance(self.checksum, str)
            or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", self.checksum) is None
        ):
            raise ValueError("pretrained file requires a Hub content checksum")


def files(values):
    if not isinstance(values, (list, tuple)) or not 2 <= len(values) <= MAX_SHARDS + 2:
        raise ValueError("pretrained initialization requires a bounded file manifest")
    result = tuple(HubFile(**value) for value in values)
    names = {file.name for file in result}
    if len(names) != len(result) or "config.json" not in names:
        raise ValueError("pretrained files require unique names and config.json")
    if INDEX not in names and names != {"config.json", "model.safetensors"}:
        raise ValueError("pretrained shards require their safetensors index")
    if not any(shard_name(name) for name in names):
        raise ValueError("pretrained manifest has no weight shards")
    return result


def disk_bytes(manifest):
    # Payloads plus bounded directory/symlink bookkeeping and temporary metadata.
    return (8 << 20) + sum(
        file.size + ((len(PurePosixPath(file.name).parts) + 3) << 12)
        for file in manifest
    )


def metadata(reference, revision, name):
    from huggingface_hub import get_hf_file_metadata, hf_hub_url

    info = get_hf_file_metadata(hf_hub_url(reference, name, revision=revision))
    if info.commit_hash != revision:
        raise ValueError("pretrained file differs from its pinned Hub revision")
    return HubFile(name, info.size, info.etag)


def _digest(file):
    if len(file.checksum) == 64:
        return hashlib.sha256()
    digest = hashlib.sha1()
    digest.update(f"blob {file.size}\0".encode())
    return digest


def verify(path, file):
    digest = _digest(file)
    with path.open("rb") as stream:
        if path.stat().st_size != file.size:
            raise ValueError(f"cached pretrained file size differs: {file.name}")
        remaining = file.size
        while remaining:
            data = stream.read(min(CHUNK_BYTES, remaining))
            if not data:
                raise ValueError(f"truncated cached pretrained file: {file.name}")
            digest.update(data)
            remaining -= len(data)
        if stream.read(1):
            raise ValueError(f"cached pretrained file grew: {file.name}")
    if digest.hexdigest() != file.checksum:
        raise ValueError(f"cached pretrained checksum differs: {file.name}")


def download(reference, revision, file, destination):
    """Stream at most the pinned size; no Xet workers or full-file buffering."""
    from huggingface_hub import hf_hub_url
    from huggingface_hub.utils import build_hf_headers, get_session, hf_raise_for_status

    digest, written = _digest(file), 0
    headers = build_hf_headers(headers={"Accept-Encoding": "identity"})
    with (
        get_session().stream(
            "GET",
            hf_hub_url(reference, file.name, revision=revision),
            headers=headers,
            follow_redirects=True,
            timeout=30,
        ) as response,
        destination.open("xb") as stream,
    ):
        hf_raise_for_status(response)
        if response.headers.get("content-encoding", "identity") != "identity":
            raise ValueError("pretrained download must use identity encoding")
        for data in response.iter_raw(chunk_size=CHUNK_BYTES):
            written += len(data)
            if written > file.size:
                raise ValueError(
                    f"pretrained download exceeds pinned size: {file.name}"
                )
            stream.write(data)
            digest.update(data)
    if written != file.size or digest.hexdigest() != file.checksum:
        raise ValueError(f"pretrained download size/checksum differs: {file.name}")


def fetch(reference, revision, file, root):
    from huggingface_hub import try_to_load_from_cache

    destination = root.joinpath(*PurePosixPath(file.name).parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cached = try_to_load_from_cache(reference, file.name, revision=revision)
    if isinstance(cached, str):
        path = Path(cached)
        verify(path, file)
        destination.symlink_to(path.resolve())
    else:
        download(reference, revision, file, destination)
    return destination


def read_index(path):
    with path.open("rb") as stream:
        payload = stream.read(INDEX_BYTES + 1)
    if len(payload) > INDEX_BYTES:
        raise ValueError("pretrained index exceeds its metadata byte limit")
    check_json(payload)
    value = json.loads(payload)
    weights = value.get("weight_map") if isinstance(value, dict) else None
    if not isinstance(weights, dict) or not 0 < len(weights) <= MAX_TENSORS:
        raise ValueError("pretrained index requires a bounded weight map")
    if any(
        not isinstance(name, str) or not 0 < len(name) <= 1024 or not shard_name(shard)
        for name, shard in weights.items()
    ):
        raise ValueError("pretrained index contains an invalid tensor or shard name")
    shards = sorted(set(weights.values()))
    if len(shards) > MAX_SHARDS:
        raise ValueError("pretrained index exceeds its shard count limit")
    return shards


def describe(reference, revision):
    """Read only config/index and HEAD metadata, never a tensor payload."""
    from huggingface_hub.errors import RemoteEntryNotFoundError

    result = [metadata(reference, revision, "config.json")]
    try:
        index = metadata(reference, revision, INDEX)
    except RemoteEntryNotFoundError:
        shards = ["model.safetensors"]
    else:
        with TemporaryDirectory(prefix="shadowspill-weight-index-") as root:
            shards = read_index(fetch(reference, revision, index, Path(root)))
        result.append(index)
    result.extend(metadata(reference, revision, name) for name in shards)
    return [asdict(file) for file in sorted(result, key=lambda file: file.name)]


@contextmanager
def snapshot(reference, revision, manifest, runtime):
    """Own temporary files per loader; external Hub caches remain read-only.

    Cleanup after each loader also bounds repeated restores of different bases.
    The node keeps the allowance/root until all workers exit, covering crashes.
    """
    if disk_bytes(manifest) > runtime.disk_bytes:
        raise ValueError("pretrained files exceed admitted initialization disk space")
    with TemporaryDirectory(prefix="weights-", dir=runtime.cache_root) as root:
        root = Path(root)
        for file in manifest:
            fetch(reference, revision, file, root)
        yield root
