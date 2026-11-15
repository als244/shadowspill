"""Installed NumPy-only demo client: save, reconnect, inspect and reuse Plans."""

import importlib.util
import json
import socket
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from shadowspill_client import SSH, ServiceClient
from shadowspill_client.protocol.training import ResolvedTraining

raw = sys.stdin.buffer.read((1 << 20) + 1)
assert len(raw) <= 1 << 20
options = json.loads(raw)
absent = {
    name: importlib.util.find_spec(name) is None
    for name in ("torch", "triton", "shadowspill", "shadowspill_workloads")
}
assert all(absent.values())
config = ResolvedTraining.from_dict(options["config"])
connection = {
    "ssh": SSH(options["ssh_host"], options["port"]),
    "token": options["token"],
    "session_id": "plan-demo-session",
    "timeout": 10,
}
name = "模型 / saved + best"
with ServiceClient(**connection) as service:
    training = service.create_training_client(config=config).result(20)
    run_id = training.run_id
    saved = training.plans.save(name=name).result(10)
    reference = saved.reference
    training.close().result(10)
    saved.save(name="closed-run").result(10)
assert service._memory.used == service._work.pending.used == (0, 0)
assert service._tunnel.closed
with ServiceClient(**connection) as service:
    saved = service.get_plans(run_id, name=name).result(10)
    assert saved.reference == reference
    assert saved.summary().result(10).plans == reference
    with TemporaryDirectory(prefix="shadowspill-plan-download-") as directory:
        path = saved.download(Path(directory) / "training.ssp").result(10)
        assert path.stat().st_size == reference.size_bytes
    reused = service.create_training_client(config=config, plans=saved).result(20)
    reused.close().result(10)
assert service._memory.used == service._work.pending.used == (0, 0)
assert service._tunnel.closed
print(
    json.dumps(
        {
            "client_host": socket.gethostname(),
            "absent": absent,
            "source_run": run_id,
            "reused_run": reused.run_id,
            "plans": reference.to_dict(),
            "name": name,
            "saved_after_close": True,
            "reconnect": True,
            "verified_download": True,
            "client_resources_released": True,
        }
    )
)
