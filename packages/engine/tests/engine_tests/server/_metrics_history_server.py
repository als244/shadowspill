"""Restart real training storage with no visible GPUs and inspect, never execute."""

import importlib.abc
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split(".")[0] in {
            "torch",
            "triton",
            "pynvml",
            "shadowspill_workloads",
        }:
            raise AssertionError("framework import in metrics controller: " + fullname)


sys.meta_path.insert(0, Block())
from shadowspill.server import Allocation, Server  # noqa: E402
from shadowspill.server.authentication import BearerTokens  # noqa: E402

source, output = Path(sys.argv[1]), Path(sys.argv[2])
previous = json.loads((source / "receipt.json").read_text())
receipts = []
for run in previous["runs"]:
    phase = run["phase"]
    token = "retained-metrics-" + os.urandom(32).hex()
    with Server(
        allocation=Allocation({**os.environ, "CUDA_VISIBLE_DEVICES": ""}),
        storage_root=source / phase / "service",
        cache_root=output / (phase + "-cache"),
        authentication=BearerTokens({token: "owner"}),
    ) as server:
        server.start(port=0)
        assert not server._node.capacity.devices and server._jobs.used == (0, 0)
        result = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
                os.environ["SHADOWSPILL_DEMO_CLIENT_HOST"],
                shlex.join(
                    [
                        os.environ["SHADOWSPILL_DEMO_CLIENT_PYTHON"],
                        "-I",
                        os.environ["SHADOWSPILL_METRICS_CLIENT_SCRIPT"],
                    ]
                ),
            ],
            input=json.dumps(
                {
                    **run,
                    "port": server.address[1],
                    "token": token,
                    "ssh_host": "chicago",
                    "annotate": os.environ.get("SHADOWSPILL_METRICS_ANNOTATE") == "1",
                    "reports": os.environ.get("SHADOWSPILL_METRICS_REPORTS") == "1",
                }
            ),
            capture_output=True,
            text=True,
            timeout=180,
        )
        (output / (phase + ".stdout")).write_text(result.stdout)
        (output / (phase + ".stderr")).write_text(result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr
        record = json.loads(result.stdout)
        assert record["client_host"] == "tubingen" and record["client_claims_released"]
        assert not server._jobs.failures() and server._jobs.used == (0, 0)
        assert not server._node.used.devices and server._disk.used == (0, 0)
        node = server._node
        receipts.append(record)
    assert node.used.claims == 0
assert not any(name in sys.modules for name in ("torch", "triton", "pynvml"))
(output / "receipt.json").write_text(
    json.dumps(
        {
            "source": str(source),
            "runs": receipts,
            "no_gpu_or_training_workers": True,
            "all_claims_released": True,
            "controller_framework_free": True,
        },
        indent=2,
    )
    + "\n"
)
