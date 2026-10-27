"""Shared worker descriptions carry configuration, never live backend pointers."""

import json

import pytest

from benchmarking.spill import record, resolve
from shadowspill.memory import PinnedHostPool
from shadowspill.network import RemotePool
from shadowspill.ssd import SSDPool


@pytest.mark.parametrize("kind", ["pinned", "ssd", "remote"])
def test_configuration_roundtrip(tmp_path, kind):
    pool = {
        "pinned": lambda: PinnedHostPool(1 << 30),
        "ssd": lambda: SSDPool(
            1 << 30,
            directory=tmp_path,
            staging_bytes=96 << 20,
            chunk_bytes=4 << 20,
            queue_depth=8,
        ),
        "remote": lambda: RemotePool(1 << 30, host="unavailable", port=19426),
    }[kind]()
    request = json.loads(json.dumps({"spill": record(pool)}))
    assert resolve(request) == pool
