"""Bounded pretrained file manifests, checksums and safetensors index metadata."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from shadowspill.checkpoint.inspection import check_json

CONFIG_BYTES = 1 << 20

INDEX_BYTES = 32 << 20

MAX_SHARDS = 512

MAX_TENSORS = 262144

CHUNK_BYTES = 1 << 20

INDEX = "model.safetensors.index.json"
QUANT_CONFIG = "hf_quant_config.json"


def quantization_file(metadata):
    if metadata.get("quantization_config"):
        from shadowspill_workloads.models.families import checkpoint_families

        text = metadata.get("text_config", metadata)
        if not isinstance(text, dict):
            raise ValueError("pretrained text_config must be a JSON object")
        kind = text.get("model_type", metadata.get("model_type"))
        family = checkpoint_families().get(kind)
        if family:
            return family.quantization_file
    return None


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
class WeightFile:
    name: str
    size: int
    checksum: str

    def __post_init__(self):
        if self.name not in ("config.json", QUANT_CONFIG, INDEX) and not shard_name(
            self.name
        ):
            raise ValueError("invalid pretrained file name")
        if type(self.size) is not int or self.size <= 0:
            raise ValueError("pretrained file size must be positive")
        limit = {
            "config.json": CONFIG_BYTES,
            QUANT_CONFIG: CONFIG_BYTES,
            INDEX: INDEX_BYTES,
        }.get(self.name)
        if limit is not None and self.size > limit:
            raise ValueError("pretrained JSON exceeds its metadata byte limit")
        if (
            not isinstance(self.checksum, str)
            or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", self.checksum) is None
        ):
            raise ValueError("pretrained file requires a Hub content checksum")


def files(values):
    if not isinstance(values, (list, tuple)) or not 2 <= len(values) <= MAX_SHARDS + 3:
        raise ValueError("pretrained initialization requires a bounded file manifest")
    result = tuple(WeightFile(**value) for value in values)
    names = {file.name for file in result}
    if len(names) != len(result) or "config.json" not in names:
        raise ValueError("pretrained files require unique names and config.json")
    if INDEX not in names and names - {QUANT_CONFIG} != {
        "config.json",
        "model.safetensors",
    }:
        raise ValueError("pretrained shards require their safetensors index")
    if not 1 <= sum(shard_name(name) for name in names) <= MAX_SHARDS:
        raise ValueError("pretrained manifest has an invalid shard count")
    return result


def disk_bytes(manifest):
    # Payloads plus bounded directory/symlink bookkeeping and temporary metadata.
    return (8 << 20) + sum(
        file.size + ((len(PurePosixPath(file.name).parts) + 3) << 12)
        for file in manifest
    )


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
