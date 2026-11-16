"""Bound checkpoint metadata before the existing safetensors reader opens it."""

import json
import struct

from shadowspill.checkpoint.inspection import check_json

from .hub import read_config
from .weight_files import INDEX, MAX_TENSORS, read_index, shard_name

HEADER_BYTES = 4 << 20


def validate(directory, manifest):
    read_config(directory / "config.json")
    shards = sorted(file.name for file in manifest if shard_name(file.name))
    if (
        any(file.name == INDEX for file in manifest)
        and read_index(directory / INDEX) != shards
    ):
        raise ValueError("checkpoint index differs from the pinned shard manifest")
    headers = tensors = 0
    for name in shards:
        path = directory / name
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError("truncated safetensors header")
            length = struct.unpack("<Q", prefix)[0]
            headers += length
            if headers > HEADER_BYTES or length > path.stat().st_size - 8:
                raise ValueError("safetensors headers exceed the metadata byte limit")
            payload = stream.read(length)
        check_json(payload)
        header = json.loads(payload)
        if not isinstance(header, dict):
            raise ValueError("safetensors header must be a JSON object")
        for key, value in header.items():
            if key == "__metadata__":
                _metadata(value)
                continue
            tensors += 1
            if tensors > MAX_TENSORS:
                raise ValueError("safetensors headers exceed the tensor count limit")
            _tensor(key, value)


def _metadata(value):
    if (
        not isinstance(value, dict)
        or len(value) > 128
        or any(
            not isinstance(key, str)
            or not isinstance(text, str)
            or len(key) > 1024
            or len(text) > 1024
            for key, text in value.items()
        )
    ):
        raise ValueError("safetensors metadata exceeds its text/count bounds")


def _tensor(name, value):
    if (
        not isinstance(name, str)
        or not 0 < len(name) <= 1024
        or not isinstance(value, dict)
        or set(value) != {"dtype", "shape", "data_offsets"}
        or not isinstance(value["dtype"], str)
        or len(value["dtype"]) > 16
    ):
        raise ValueError("invalid bounded safetensors tensor declaration")
    for key, limit in (("shape", 8), ("data_offsets", 2)):
        sequence = value[key]
        if (
            not isinstance(sequence, list)
            or len(sequence) > limit
            or any(
                type(size) is not int or not 0 <= size < (1 << 63) for size in sequence
            )
        ):
            raise ValueError("safetensors tensor dimensions exceed their bounds")
    # The existing reader validates dtypes, exact extents and index membership.
