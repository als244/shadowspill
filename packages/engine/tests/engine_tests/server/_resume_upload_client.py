"""NumPy-only client: interrupted uploads and explicit checkpoint storage copies."""

import hashlib
import importlib.util
import json
import socket
import sys
from pathlib import Path
from tempfile import mkdtemp

from shadowspill_client import SSH, ServiceClient
from shadowspill_client.protocol.requests import ArtifactRef
from shadowspill_client.transport.http import TransportError

options = json.loads(sys.stdin.buffer.read((1 << 20) + 1))
absent = {
    name: importlib.util.find_spec(name) is None
    for name in ("torch", "triton", "shadowspill", "shadowspill_workloads")
}
assert all(absent.values())
directory = Path(mkdtemp(prefix="shadowspill-resumed-upload-"))
path = directory / "input.bin"
path.write_bytes(bytes(range(256)) * 32769)
checkpoint = ArtifactRef.from_dict(options["checkpoint"])
dependencies = [ArtifactRef.from_dict(value) for value in options["dependencies"]]


def connect(port):
    return ServiceClient(
        ssh=SSH(options["ssh_host"], port), token=options["token"], max_connections=1
    )


with connect(options["source_port"]) as source:
    source.artifact(checkpoint).download(directory / "checkpoint").result(30)
    for reference in dependencies:
        source.artifact(reference).download(directory / reference.id).result(30)
assert source._memory.used == source._work.pending.used == (0, 0)

with connect(options["port"]) as first:
    try:
        first.artifacts.upload(path).result(30)
    except TransportError:
        pass
    else:
        raise AssertionError("expected the injected interrupted upload")
assert first._memory.used == first._work.pending.used == (0, 0)

with connect(options["port"]) as second:
    handle = second.artifacts.upload(path).result(30)
    expected = ArtifactRef.from_dict(options["reference"])
    assert handle.reference == expected
    copy = handle.download(directory / "roundtrip.bin").result(30)
    with copy.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == expected.sha256
    uploaded = second.artifacts.upload(
        directory / "checkpoint", reference=checkpoint
    ).result(30)
    manifest = uploaded.manifest().result(30)
    assert uploaded.reference == checkpoint
    assert manifest.info.completed_updates == 9
    for reference in dependencies:
        copied = second.artifacts.upload(
            directory / reference.id, reference=reference
        ).result(30)
        assert copied.reference == reference
    assert (
        second.artifacts.upload(directory / "checkpoint", reference=checkpoint).result(
            30
        )
        == uploaded
    )
assert second._memory.used == second._work.pending.used == (0, 0)
assert all(importlib.util.find_spec(name) is None for name in absent)
print(
    json.dumps(
        {
            "client_host": socket.gethostname(),
            "absent": absent,
            "bytes": expected.size_bytes,
            "sha256": expected.sha256,
            "directory": str(directory),
            "checkpoint": checkpoint.to_dict(),
            "rank_count": len(manifest.ranks),
            "completed_updates": manifest.info.completed_updates,
            "all_connections_released": True,
        }
    )
)
