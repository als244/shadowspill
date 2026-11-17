"""Run on the demo laptop with only NumPy and the installed client wheel."""

import importlib.util
import json
import socket
import sys

from shadowspill_client import SSH, ServiceClient

raw = sys.stdin.buffer.read((1 << 20) + 1)
assert len(raw) <= 1 << 20
options = json.loads(raw)
absent = {
    name: importlib.util.find_spec(name) is None
    for name in ("torch", "triton", "shadowspill", "shadowspill_workloads")
}
assert all(absent.values())
with ServiceClient(
    ssh=SSH(options["ssh_host"], options["port"]),
    token=options["token"],
    timeout=10,
) as client:
    inventory = client.list_hardware().result(20)
    assert inventory.nodes and "inspection reserves no hardware" in inventory.summary()
    preview = client.describe_training(**options["setup"]).result(20)
    assert preview.config.model.initialize is None
    assert preview.config.optimizer.defaults == {"lr": 0.01}
    assert client.describe_training(config=preview.config).result(20) == preview
    assert preview.hardware and preview.resources is not None
    assert "no measured timings or resource reservation" in preview.summary()
assert client._memory.used == client._work.pending.used == (0, 0)
assert client._tunnel.closed
print(
    json.dumps(
        {
            "client_host": socket.gethostname(),
            "public_service_client": True,
            "preview_roundtrip": True,
            "hardware_inventory": inventory.to_dict(),
            "absent": absent,
            "estimates": [item.to_dict() for item in preview.preparation],
            "client_resources_released": True,
        }
    )
)
