"""Isolated laptop transfer resume across distinct owned SSH connections."""

import hashlib
import importlib.util
import json
import socket
import sys
from pathlib import Path
from tempfile import mkdtemp

from shadowspill_client import SSH, ServiceClient
from shadowspill_client.downloads import partial_path
from shadowspill_client.protocol.codec import ProtocolError
from shadowspill_client.protocol.requests import ArtifactRef
from shadowspill_client.transport.http import TransportError

options = json.loads(sys.stdin.buffer.read((1 << 20) + 1))
reference = ArtifactRef.from_dict(options["reference"])
absent = {
    name: importlib.util.find_spec(name) is None
    for name in ("torch", "triton", "shadowspill", "shadowspill_workloads")
}
assert all(absent.values())
destination = Path(mkdtemp(prefix="shadowspill-resumed-download-")) / "artifact.bin"

with ServiceClient(
    ssh=SSH(options["ssh_host"], options["port"]),
    token=options["token"],
    max_connections=1,
) as first:
    try:
        first.artifact(reference).download(destination, resume=True).result(30)
    except (ProtocolError, TransportError):
        pass
    else:
        raise AssertionError("expected the injected interrupted stream")
    assert not destination.exists()
    offset = (partial_path(destination) / "data").stat().st_size
    assert offset == options["cutoff"]
assert first._memory.used == first._work.pending.used == (0, 0)

with ServiceClient(
    ssh=SSH(options["ssh_host"], options["port"]),
    token=options["token"],
    max_connections=1,
) as second:
    second.artifact(reference).download(destination, resume=True).result(30)
    assert destination.stat().st_size == reference.size_bytes
    with destination.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == reference.sha256
    assert not partial_path(destination).exists()
assert second._memory.used == second._work.pending.used == (0, 0)
print(
    json.dumps(
        {
            "client_host": socket.gethostname(),
            "absent": absent,
            "verified_offset": offset,
            "bytes": reference.size_bytes,
            "sha256": reference.sha256,
            "destination": str(destination),
            "both_connections_released": True,
        }
    )
)
