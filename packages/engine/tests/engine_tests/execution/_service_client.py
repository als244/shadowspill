"""NumPy-only fixture client, runnable locally or on a separate demo machine."""

import importlib.abc
import json
import socket
import sys
from contextlib import ExitStack
from io import BytesIO
from threading import Event
from time import monotonic


class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {
            "torch",
            "triton",
            "pynvml",
            "shadowspill",
            "shadowspill_workloads",
        }:
            raise AssertionError("engine/framework import in client: " + fullname)


sys.meta_path.insert(0, Block())

import numpy as np  # noqa: E402

from shadowspill_client import SSH, Datum  # noqa: E402
from shadowspill_client.buffers import BufferBudget  # noqa: E402
from shadowspill_client.protocol.creation import TrainingCreation  # noqa: E402
from shadowspill_client.protocol.payload import MEDIA_TYPE, snapshot_batch  # noqa: E402
from shadowspill_client.protocol.payload_io import read_payload  # noqa: E402
from shadowspill_client.protocol.requests import (  # noqa: E402
    ArtifactRef,
    OperationRequest,
    RequestIdentity,
)
from shadowspill_client.protocol.training import ResolvedTraining  # noqa: E402
from shadowspill_client.transport.http import HTTPTransport  # noqa: E402
from shadowspill_client.transport.ssh import SSHTunnel  # noqa: E402


def records(targets):
    return [
        Datum(
            {"pixels": np.arange(4, dtype=np.float32).reshape(1, 2, 2) / 8 + i / 4},
            {"class_id": i, "weights": weight, "valid": True} if targets else {},
        )
        for i, weight in enumerate((1.0, -0.5, 0.0))
    ]


def ready(client, route, *, state="ready"):
    deadline = monotonic() + 180
    while True:
        reply = client.request("GET", route)
        assert reply.value.state not in ("failed", "stopped"), reply.value.error
        if reply.status == 200 and reply.value.state == state:
            return reply.value
        assert monotonic() < deadline, "creation exceeded its integration deadline"
        Event().wait(0.05)


def qualify(client, config, memory, oracle):
    saved, receipts = None, []
    service = client.request("GET", "/v1/info").value
    for mode in ("fresh", "saved"):
        route = "/v1/runs/factory-session/" + mode
        creation = TrainingCreation(config, plans=saved, name="demo-" + mode)
        client.request("POST", route, creation)
        info = ready(client, route)
        assert info.plans is not None and info.owner_id == service.owner_id
        assert client.request("POST", route, creation).value == info
        saved = info.plans
        writer = client.request("POST", route + "/writers/demo-writer").value

        def invoke(
            operation,
            sequence,
            *,
            output=None,
            run_id=mode,
            base=route,
            lease=writer,
            **options,
        ):
            name, payload, wire, count = f"demo-{sequence}", None, None, 0
            if operation != "close":
                datums = records(operation != "forward")
                with snapshot_batch(datums, memory) as snapshot:
                    payload = ArtifactRef(
                        name, snapshot.sha256, snapshot.size_bytes, MEDIA_TYPE
                    )
                    wire, count = b"".join(snapshot.chunks()), len(datums)
            request = OperationRequest(
                RequestIdentity(
                    name,
                    run_id,
                    sequence,
                    lease.writer_id,
                    lease.writer_epoch,
                    service.server_incarnation,
                ),
                operation,
                payload,
                count,
                **({} if output is None else {"outputs": output}),
                **options,
            )
            client.request("POST", base + "/requests", request)
            path = base + "/requests/" + name
            accepted = (
                client.request("PUT", path + "/input")
                if wire is None
                else client.upload(path + "/input", payload, [wire])
            ).value
            assert accepted.acceptance is not None, accepted
            deadline = monotonic() + 120
            while True:
                status = client.request(
                    "GET", path + "?wait=completed&timeout=10"
                ).value
                if status.state in ("completed", "skipped", "failed"):
                    break
                assert monotonic() < deadline
            assert status.state == "completed", status
            # An identical request after completion queries the original effect.
            assert client.request("POST", base + "/requests", request).value == status
            return status.result

        result = invoke(
            "step",
            1,
            optim={"lr": 0.03},
            grad_scale=0.5,
            loss_params={"temperature": 2.0},
        )
        np.testing.assert_allclose(result.loss, oracle["loss"], rtol=1e-4)
        prediction = invoke("forward", 2, output="predictions")
        reference = prediction.outputs["prediction"]
        closed = invoke("close", 3)
        assert closed.completed_updates == 1
        assert ready(client, route, state="closed").plans == saved
        assert client.request("POST", route, creation).value.state == "closed"
        # Results remain downloadable after both execution and creation retire.
        target = BytesIO()
        client.download("/v1/artifacts/" + reference.id, reference, target)
        with read_payload(BytesIO(target.getvalue()), reference, memory) as decoded:
            rows = decoded.datums[0].inputs
            actual = np.stack([rows[str(index)].array() for index in range(3)])
            np.testing.assert_allclose(
                actual, oracle["after_step"], rtol=1e-4, atol=1e-5
            )
        receipts.append(
            {
                "mode": mode,
                "completed_updates": closed.completed_updates,
                "plans": saved.to_dict(),
            }
        )
    page = client.request("GET", "/v1/runs/factory-session").value
    assert [row.run_id for row in page.runs] == ["fresh", "saved"]
    assert all(row.state == "closed" for row in page.runs)
    return receipts


raw = sys.stdin.buffer.read((1 << 20) + 1)
assert len(raw) <= 1 << 20
options = json.loads(raw)
memory = BufferBudget(64 << 20, 8)
with ExitStack() as stack:
    base = f"http://127.0.0.1:{options['port']}"
    if options.get("ssh_host"):
        tunnel = stack.enter_context(
            SSHTunnel(SSH(options["ssh_host"], options["port"]), memory=memory)
        )
        base = tunnel.start()
    client = HTTPTransport(base, token=options["token"], memory=memory, timeout=30)
    receipts = qualify(
        client, ResolvedTraining.from_dict(options["config"]), memory, options["oracle"]
    )
assert memory.used == (0, 0)
assert not any(name in sys.modules for name in ("torch", "triton", "shadowspill"))
print(
    json.dumps(
        {
            "client_host": socket.gethostname(),
            "python": sys.version,
            "numpy": np.__version__,
            "runs": receipts,
            "all_client_claims_released": True,
        }
    )
)
