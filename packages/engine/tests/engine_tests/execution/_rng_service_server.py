"""Framework-free public server for the isolated stochastic demo client."""

import importlib.abc
import json
import os
import shlex
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from threading import Event
from time import monotonic


class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split(".")[0] in {
            "torch",
            "triton",
            "pynvml",
            "shadowspill_workloads",
        }:
            raise AssertionError("GPU/catalog import in controller: " + fullname)


sys.meta_path.insert(0, Block())

from shadowspill.server import Allocation, Server, ServerLimits  # noqa: E402
from shadowspill.server.authentication import BearerTokens  # noqa: E402

root = Path(sys.argv[1])
options = json.loads((root / "setup.json").read_text())
token = "stochastic-" + os.urandom(32).hex()
limits = ServerLimits()
limits = replace(
    limits,
    max_http_requests=2,
    max_open_records=1,
    max_artifact_owners=1,
    artifact_bytes=256 << 20,
    artifact_count=1000,
    temporary_disk_bytes=512 << 20,
    temporary_files=300,
    creation=replace(
        limits.creation, preparation_disk_bytes=256 << 20, log_bytes=64 << 10
    ),
)
with Server(
    allocation=Allocation(
        {
            **os.environ,
            "TORCHINDUCTOR_CACHE_DIR": str(root / "inductor"),
            "TRITON_CACHE_DIR": str(root / "triton"),
        }
    ),
    storage_root=root / "service",
    cache_root=root / "cache",
    workload_packages=["external-classifier-fixture"],
    authentication=BearerTokens({token: "owner"}),
    limits=limits,
) as service:
    service.start(port=0)
    options.update(port=service.address[1], token=token)
    host = os.environ.get("SHADOWSPILL_DEMO_CLIENT_HOST")
    if host:
        command = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            host,
            shlex.join(
                [
                    os.environ["SHADOWSPILL_DEMO_CLIENT_PYTHON"],
                    "-I",
                    os.environ["SHADOWSPILL_RNG_CLIENT_SCRIPT"],
                ]
            ),
        ]
        options["ssh_host"] = "chicago"
    else:
        command = [
            sys.executable,
            str(Path(__file__).with_name("_rng_service_client.py")),
        ]
    client = subprocess.run(
        command, input=json.dumps(options), text=True, capture_output=True, timeout=600
    )
    (root / "client.stderr").write_text(client.stderr)
    assert client.returncode == 0, client.stdout + client.stderr
    receipt = json.loads(client.stdout)
    if host:
        assert receipt["client_host"] == "tubingen"
    assert receipt["all_client_claims_released"]
    deadline = monotonic() + 30
    while service._jobs.used != (0, 0):
        assert monotonic() < deadline, service._jobs.failures()
        Event().wait(0.02)
    assert not service._jobs.failures()
    assert not service._node.used.devices
    assert service._node.used.host_bytes == limits.fixed_host_bytes
    assert service._disk.used == (0, 0)
    node = service._node
assert node.used.claims == 0
assert not any(name in sys.modules for name in ("torch", "triton", "pynvml"))
receipt.update(
    all_server_claims_released=True,
    controller_framework_free=True,
    model="two-layer tanh MLP with multiplicative uniform noise",
    oracle="independent NumPy forward/loss/SGD derivatives",
    rtol=2e-4,
    atol=2e-5,
)
(root / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(
    "Qualified stochastic public step, split updates, native/converted resume "
    "and evaluation isolation"
)
