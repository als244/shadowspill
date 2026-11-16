"""Sequential public services: no live workers or storage reused after transfer."""

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
token = "upload-training-" + os.urandom(32).hex()
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
receipts = []
for phase in ("source", "target"):
    directory = root / phase
    with Server(
        allocation=Allocation(
            {
                **os.environ,
                "TORCHINDUCTOR_CACHE_DIR": str(root / "inductor"),
                "TRITON_CACHE_DIR": str(root / "triton"),
            }
        ),
        service_root=directory / "service",
        cache_root=directory / "cache",
        workload_packages=["external-classifier-fixture"],
        authentication=BearerTokens({token: "owner"}),
        limits=limits,
    ) as service:
        service.start(port=0)
        with service._storage.artifacts("owner") as artifacts:
            assert not artifacts.page()  # Target really starts with empty storage.
        options.update(port=service.address[1], token=token, phase=phase)
        if receipts:
            options["backup"] = receipts[0]["backup"]
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
                        os.environ["SHADOWSPILL_UPLOAD_TRAINING_CLIENT_SCRIPT"],
                    ]
                ),
            ]
            options["ssh_host"] = "chicago"
        else:
            command = [
                sys.executable,
                str(Path(__file__).with_name("_upload_service_client.py")),
            ]
        result = subprocess.run(
            command,
            input=json.dumps(options),
            text=True,
            capture_output=True,
            timeout=360,
        )
        (directory / "client.stdout").write_text(result.stdout)
        (directory / "client.stderr").write_text(result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr
        receipt = json.loads(result.stdout)
        assert receipt["all_client_claims_released"]
        if host:
            assert receipt["client_host"] == "tubingen"
        deadline = monotonic() + 30
        while service._jobs.used != (0, 0):
            assert monotonic() < deadline, service._jobs.failures()
            Event().wait(0.02)
        assert not service._jobs.failures()
        assert not service._node.used.devices
        assert service._node.used.host_bytes == limits.fixed_host_bytes
        assert service._disk.used == (0, 0)
        node = service._node
        receipts.append(receipt)
    assert node.used.claims == 0
assert not any(name in sys.modules for name in ("torch", "triton", "pynvml"))
(root / "receipt.json").write_text(
    json.dumps(
        {
            "runs": receipts,
            "all_server_claims_released": True,
            "controller_framework_free": True,
            "target_storage_started_empty": True,
            "model": "two-layer tanh MLP with multiplicative uniform noise",
            "oracle": "independent NumPy forward/loss/SGD derivatives",
            "rtol": 2e-4,
            "atol": 2e-5,
        },
        indent=2,
    )
    + "\n"
)
