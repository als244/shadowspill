"""Public server fixture; only its build/execution children import frameworks."""

import importlib.abc
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from threading import Event
from time import monotonic


class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split(".")[0] in {"torch", "triton", "pynvml"}:
            raise AssertionError("framework import in controller: " + fullname)
        if fullname.split(".")[0] in {"benchmarks", "reference", "qualification"} or (
            fullname.startswith("shadowspill_workloads.")
            and ".implementations." in fullname
        ):
            raise AssertionError("model import in controller: " + fullname)


sys.meta_path.insert(0, Block())

from shadowspill.server import Allocation, Server, ServerLimits  # noqa: E402
from shadowspill.server.authentication import BearerTokens  # noqa: E402

root = Path(sys.argv[1])
options = json.loads((root / "setup.json").read_text())
if (
    options.get("fresh_saved_plans")
    or options.get("weights_initialization")
    or options.get("plan_first")
):
    import shadowspill.server.startup.creation as creation

    original_prepare = creation.prepare_job
    preparations = 0

    def prepare_once(*args, **kwargs):
        global preparations
        preparations += 1
        assert preparations == 1, "saved Plans unexpectedly restarted preparation"
        return original_prepare(*args, **kwargs)

    creation.prepare_job = prepare_once
if options.get("local_planning"):
    import shadowspill.planning.plan as planner
    import shadowspill.server.startup.creation as creation

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "local Plan submission attempted server preparation/search"
        )

    creation.prepare_job = forbidden
    planner.plan_program = forbidden
if options.get("guard_measurement_reuse"):
    import shadowspill.training.preparation.phases.launcher as preparation_phases

    def measured_command(module, arguments, *, ranks=1):
        assert ranks == 1
        return [
            sys.executable,
            str(Path(__file__).with_name("_measurement_phase_worker.py")),
            module,
            *map(str, arguments),
        ]

    preparation_phases.worker_command = measured_command
if options.get("cold_llama_workers") or options.get("cold_workers"):
    import shadowspill.training.worker.launch as worker_launch

    def cold_command(module, arguments):
        assert module == "shadowspill.training.worker.bootstrap"
        # Only these fixture-owned caches are removed, after preparation exits.
        for name in ("inductor", "triton"):
            shutil.rmtree(root / name, ignore_errors=True)
        return [
            sys.executable,
            *(["-S"] if sys.flags.no_site else []),
            str(
                Path(__file__).with_name(
                    "_saved_service_worker.py"
                    if options.get("cold_workers")
                    else "_llama_cold_worker.py"
                )
            ),
            *map(str, arguments),
        ]

    worker_launch.worker_command = cold_command
token = "regression-" + os.urandom(32).hex()
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
    service_root=root / "service",
    cache_root=root / "cache",
    workload_packages=["shadowspill", "shadowspill-workloads"],
    authentication=BearerTokens({token: "owner"}),
    limits=limits,
) as service:
    service.start(port=0)
    options.update(port=service.address[1], token=token)
    host = os.environ.get("SHADOWSPILL_DEMO_CLIENT_HOST")
    if host:
        remote = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            host,
        ]
        client_python = os.environ["SHADOWSPILL_DEMO_CLIENT_PYTHON"]
        expected_client_host = subprocess.check_output(
            [
                *remote,
                shlex.join(
                    [
                        client_python,
                        "-I",
                        "-c",
                        "import socket; print(socket.gethostname())",
                    ]
                ),
            ],
            text=True,
            timeout=30,
        ).strip()
        command = [
            *remote,
            shlex.join(
                [
                    client_python,
                    "-I",
                    os.environ["SHADOWSPILL_DEMO_CLIENT_SCRIPT"],
                ]
            ),
        ]
        options["ssh_host"] = os.environ.get(
            "SHADOWSPILL_DEMO_SERVER_HOST", socket.gethostname()
        )
    else:
        command = [
            sys.executable,
            str(
                Path(__file__).with_name(
                    options.get("client_script", "_regression_service_client.py")
                )
            ),
        ]
    client = subprocess.run(
        command,
        input=json.dumps(options),
        text=True,
        capture_output=True,
        timeout=options.get("client_timeout", 660),
    )
    (root / "client.stderr").write_text(client.stderr)
    assert client.returncode == 0, client.stdout + client.stderr
    receipt = json.loads(client.stdout)
    if host:
        assert receipt["client_host"] == expected_client_host
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
assert not any(name in sys.modules for name in ("torch", "triton", "pynvml", "mlops"))
receipt.update(
    all_server_claims_released=True,
    controller_framework_free=True,
    model=options.get(
        "model_description", "supplied regression: Linear(16,32), SiLU, Linear(32,8)"
    ),
    oracle=options.get(
        "oracle_description", "independent NumPy SiLU derivatives and AdamW equations"
    ),
)
(root / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print("Qualified supplied regression step/split/evaluation/AdamW resume")
