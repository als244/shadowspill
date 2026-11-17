"""Remote storage inspection reads bounded, exact, revision-pinned header ranges."""

import json
import struct
from contextlib import contextmanager

import pytest

from shadowspill_workloads.initialization import headers
from shadowspill_workloads.initialization.weight_files import WeightFile


@pytest.mark.parametrize(
    "fault", [None, "status", "offset", "encoding", "long", "short"]
)
def test_hub_headers_use_exact_bounded_byte_ranges(monkeypatch, fault):
    from huggingface_hub import utils

    declarations = {
        "layer.weight": {"dtype": "BF16", "shape": [16, 32], "data_offsets": [0, 1024]}
    }
    metadata = json.dumps(declarations).encode()
    payload = struct.pack("<Q", len(metadata)) + metadata
    size = len(payload) + 1024
    requests = []

    class Response:
        status_code = 200 if fault == "status" else 206

        def __init__(self, start, stop):
            origin = start if fault != "offset" else start + 1
            self.headers = {
                "content-range": f"bytes {origin}-{stop}/{size}",
                "content-encoding": "gzip" if fault == "encoding" else "identity",
            }
            self.data = payload[start : stop + 1]

        def iter_raw(self, chunk_size):
            assert chunk_size <= 65536
            yield (
                self.data + b"extra"
                if fault == "long"
                else self.data[:-1]
                if fault == "short"
                else self.data
            )

    class Session:
        @contextmanager
        def stream(self, method, url, *, headers, **kwargs):
            assert method == "GET" and "/resolve/pinned-revision/" in url
            start, stop = map(int, headers["Range"].removeprefix("bytes=").split("-"))
            requests.append((start, stop))
            yield Response(start, stop)

    monkeypatch.setattr(utils, "get_session", Session)
    monkeypatch.setattr(utils, "hf_raise_for_status", lambda _: None)
    manifest = [WeightFile("model.safetensors", size, "0" * 64)]
    if fault is None:
        assert (
            headers.from_hub("org/model", "pinned-revision", manifest) == declarations
        )
        assert requests == [(0, 7), (8, len(payload) - 1)]
    else:
        with pytest.raises(ValueError):
            headers.from_hub("org/model", "pinned-revision", manifest)
