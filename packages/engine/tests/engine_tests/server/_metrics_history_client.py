"""NumPy-only inspection of retained GPU results through a restarted service."""

import importlib.abc
import json
import socket
import sys
from pathlib import Path
from tempfile import mkdtemp

from shadowspill_client import SSH, ServiceClient
from shadowspill_client.protocol.codec import decode
from shadowspill_client.protocol.metrics import MetricRecord, MetricSnapshot


class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split(".")[0] in {
            "torch",
            "triton",
            "shadowspill",
            "shadowspill_workloads",
        }:
            raise AssertionError("framework import in metrics client: " + fullname)


sys.meta_path.insert(0, Block())
options = json.loads(sys.stdin.buffer.read((1 << 20) + 1))
connection = {"ssh": SSH(options["ssh_host"], options["port"])}
with ServiceClient(
    **connection, token=options["token"], session_id="rng-session"
) as service:
    first = page = service.get_metrics(options["run_id"], limit=2).result(30)
    rows = list(page.records)
    while page.next_cursor is not None:
        page = service.get_metrics(
            options["run_id"], after=page.next_cursor, through=first.through, limit=2
        ).result(30)
        rows.extend(page.records)
    observed = options["observations"]
    compute = [
        row
        for row in rows
        if row.operation in ("step", "forward_backward", "optim_step", "forward")
    ]
    if options["phase"] == "source":
        compute = compute[
            1:
        ]  # Initial update precedes the retained oracle observations.
    for row, expected in zip(compute[: len(observed)], observed, strict=True):
        assert (row.operation, row.outcome, row.completed_updates, row.loss) == (
            expected["operation"],
            expected["outcome"],
            expected["updates"],
            expected["loss"],
        )
    training = service.get_training_client(options["run_id"], read_only=True).result(30)
    assert training.read_only and training.metrics.query(limit=2).result(30) == first
    for row in rows:
        completed = training.requests.get(row.identity.request_id).result(30)
        assert completed.metrics == row.metrics and completed.loss == row.loss
    destination = (
        Path(mkdtemp(prefix="shadowspill-retained-metrics-")) / "metrics.jsonl"
    )
    training.metrics.export_jsonl(destination, through=first.through, limit=2).result(
        30
    )
    lines = [decode(line) for line in destination.read_bytes().splitlines()]
    assert lines[0] == MetricSnapshot(
        "rng-session", options["run_id"], 0, first.through
    )
    assert lines[1:] == rows and all(isinstance(row, MetricRecord) for row in lines[1:])
assert service._memory.used == service._work.pending.used == (0, 0)
with ServiceClient(
    **connection, token=options["token"], session_id="rng-session"
) as again:
    assert (
        again.get_metrics(options["run_id"], after=first.through).result(30).records
        == ()
    )
assert again._memory.used == again._work.pending.used == (0, 0)
assert not any(
    name in sys.modules
    for name in ("torch", "triton", "shadowspill", "shadowspill_workloads")
)
print(
    json.dumps(
        {
            "client_host": socket.gethostname(),
            "run_id": options["run_id"],
            "records": len(rows),
            "through": first.through,
            "jsonl": str(destination),
            "qualified_losses_match": True,
            "jsonl_roundtrip": True,
            "client_claims_released": True,
        }
    )
)
