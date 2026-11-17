"""Bound checkpoint metadata before the existing safetensors reader opens it."""

import json
import struct

from shadowspill.checkpoint.inspection import check_json

from .hub import read_config
from .weight_files import INDEX, MAX_TENSORS, read_index, shard_name

HEADER_BYTES = 32 << 20


def validate(directory, manifest):
    read(directory, manifest)


def read(directory, manifest):
    """Read bounded tensor declarations without constructing tensors or payloads."""
    read_config(directory / "config.json")
    shards = sorted(file.name for file in manifest if shard_name(file.name))
    if (
        any(file.name == INDEX for file in manifest)
        and read_index(directory / INDEX) != shards
    ):
        raise ValueError("checkpoint index differs from the pinned shard manifest")

    def read_range(name, offset, length):
        with (directory / name).open("rb") as stream:
            stream.seek(offset)
            return stream.read(length)

    return _read_shards(manifest, read_range)


def from_hub(reference, revision, manifest):
    """Read only pinned shard headers; a server ignoring Range is rejected."""
    from huggingface_hub import hf_hub_url
    from huggingface_hub.utils import build_hf_headers, get_session, hf_raise_for_status

    sizes = {file.name: file.size for file in manifest}

    def read_range(name, offset, length):
        headers = build_hf_headers(
            headers={
                "Accept-Encoding": "identity",
                "Range": f"bytes={offset}-{offset + length - 1}",
            }
        )
        with get_session().stream(
            "GET",
            hf_hub_url(reference, name, revision=revision),
            headers=headers,
            follow_redirects=True,
            timeout=30,
        ) as response:
            hf_raise_for_status(response)
            if response.status_code != 206:
                raise ValueError(
                    "checkpoint header request requires byte-range support"
                )
            if response.headers.get("content-encoding", "identity") != "identity":
                raise ValueError("checkpoint headers require identity encoding")
            if response.headers.get("content-range") != (
                f"bytes {offset}-{offset + length - 1}/{sizes[name]}"
            ):
                raise ValueError(
                    "checkpoint header response differs from requested range"
                )
            content = bytearray()
            for chunk in response.iter_raw(chunk_size=65536):
                if len(content) + len(chunk) > length:
                    raise ValueError(
                        "checkpoint header response exceeds requested range"
                    )
                content.extend(chunk)
            return bytes(content)

    return _read_shards(manifest, read_range)


def _read_shards(manifest, read_range):
    headers = 0
    result = {}
    shards = {file.name: file for file in manifest if shard_name(file.name)}
    for name in shards:
        prefix = read_range(name, 0, 8)
        if len(prefix) != 8:
            raise ValueError("truncated safetensors header")
        length = struct.unpack("<Q", prefix)[0]
        headers += length
        if headers > HEADER_BYTES or length > shards[name].size - 8:
            raise ValueError("safetensors headers exceed the metadata byte limit")
        payload = read_range(name, 8, length)
        if len(payload) != length:
            raise ValueError("truncated safetensors header")
        check_json(payload)
        header = json.loads(payload)
        if not isinstance(header, dict):
            raise ValueError("safetensors header must be a JSON object")
        for key, value in header.items():
            if key == "__metadata__":
                _metadata(value)
                continue
            if key in result:
                raise ValueError("duplicate checkpoint tensor name")
            if len(result) >= MAX_TENSORS:
                raise ValueError("safetensors headers exceed the tensor count limit")
            _tensor(key, value)
            result[key] = value
    return result


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
