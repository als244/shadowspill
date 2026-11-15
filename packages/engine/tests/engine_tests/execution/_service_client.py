"""NumPy-only fixture client, runnable locally or on a separate demo machine."""

import hashlib
import importlib.abc
import json
import socket
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
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

from shadowspill_client import SSH, Datum, ServiceClient  # noqa: E402
from shadowspill_client.buffers import BufferBudget  # noqa: E402
from shadowspill_client.protocol.codec import decode, digest  # noqa: E402
from shadowspill_client.protocol.payload_io import read_payload  # noqa: E402
from shadowspill_client.protocol.training import ResolvedTraining  # noqa: E402


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


def split_update(training):
    training.forward_backward(records(True), loss_params={"temperature": 0.5}).result(
        120
    )
    training.forward_backward(records(True)).result(120)
    return training.optim_step(optim={"lr": 0.02}, grad_scale=-0.5).result(120)


def qualify(client, config, memory, oracle):
    saved, receipts = None, []
    preview = client.describe_training(config=config).result(20)
    assert preview.config == config
    for mode in ("fresh", "saved"):
        pending = client.create_training_client(
            config=preview.config, plans=saved, name="demo-" + mode
        )
        assert pending.run_id
        training = pending.result(180)
        assert training.run_id == pending.run_id and training.get_config() == config
        saved = training.plans
        assert saved is not None
        summary, index = [], 0
        while True:
            page = saved.summary(page=index).result(10)
            assert page.plans == saved.reference
            assert page.configuration_sha256 == digest(config)
            summary.append(page)
            if page.next_page is None:
                break
            index = page.next_page
        summary_rows = [row for page in summary for row in page.rows]
        planned = [row for row in summary_rows if row.kind == "worker_plan"]
        assert len(planned) == page.plan_count and page.plan_count > 0
        assert all(row.values["simulated"] for row in planned)
        assert all(row.values["duration_seconds"] >= 0 for row in planned)
        assert {row.kind for row in summary_rows} == {
            "configuration",
            "batch",
            "build_case",
            "representative",
            "candidate",
            "worker_plan",
            "coverage",
            "shared_memory",
        }
        step = training.step(
            records(True),
            optim={"lr": 0.03},
            grad_scale=0.5,
            loss_params={"temperature": 2.0},
        )
        assert step.request_id and step.accepted().result(30).identity == step.identity
        result = step.result(120)
        np.testing.assert_allclose(result.loss, oracle["loss"], rtol=1e-4)
        assert training.requests.get(step.request_id).result(10).record == result.record
        started = monotonic()
        checkpoint_future = training.save_checkpoint(
            name="step-1",
            metadata={"data_cursor": 3, "preparation": mode},
            weights="master" if mode == "fresh" else "compute",
        )
        assert (
            checkpoint_future.accepted().result(30).identity
            == checkpoint_future.identity
        )
        checkpoint = checkpoint_future.result(120)
        save_seconds = monotonic() - started
        checkpoint_manifest = checkpoint.manifest().result(30)
        assert checkpoint_manifest.info.completed_updates == 1
        assert checkpoint_manifest.info.request_id == checkpoint_future.request_id
        assert checkpoint_manifest.info.metadata["data_cursor"] == 3
        assert checkpoint_manifest.info.plans == saved.reference
        assert checkpoint_manifest.info.configuration.sha256 == digest(config)

        assert split_update(training).completed_updates == 2
        restore_started = monotonic()
        restore = training.load_checkpoint(
            "step-1" if mode == "fresh" else checkpoint.id
        )
        assert restore.accepted().result(30).checkpoint == checkpoint.reference
        restored = restore.result(120)
        restore_seconds = monotonic() - restore_started
        assert restored.completed_updates == 1
        assert restored.history_id != result.history_id
        assert restored.metadata["data_cursor"] == 3
        assert restored.checkpoint == checkpoint
        assert training.get_config() == config
        prediction = training.forward(records(False), outputs="predictions").result(120)
        output = prediction.outputs["prediction"]
        resumed_update = split_update(training)
        assert resumed_update.completed_updates == 2
        assert resumed_update.history_id == restored.history_id
        resumed_prediction = training.forward(
            records(False), outputs="predictions"
        ).result(120)
        closed = training.close().result(30)
        assert closed.completed_updates == 2
        route = "/v1/runs/factory-session/" + training.run_id
        assert ready(client._transport, route, state="closed").plans == saved.reference
        # Public output handles download only on explicit request, after worker exit.
        with TemporaryDirectory(prefix="shadowspill-demo-output-") as directory:
            path = output.download(Path(directory) / "predictions.ssp").result(30)
            with (
                path.open("rb") as stream,
                read_payload(stream, output.reference, memory) as decoded,
            ):
                rows = decoded.datums[0].inputs
                actual = np.stack([rows[str(index)].array() for index in range(3)])
                np.testing.assert_allclose(
                    actual, oracle["after_step"], rtol=1e-4, atol=1e-5
                )
        resumed_output = resumed_prediction.outputs["prediction"]
        with TemporaryDirectory(prefix="shadowspill-demo-restored-") as directory:
            path = resumed_output.download(Path(directory) / "predictions.ssp").result(
                30
            )
            with (
                path.open("rb") as stream,
                read_payload(stream, resumed_output.reference, memory) as decoded,
            ):
                rows = decoded.datums[0].inputs
                actual = np.stack([rows[str(index)].array() for index in range(3)])
                np.testing.assert_allclose(
                    actual, oracle["after_split"], rtol=1e-4, atol=1e-5
                )
        with TemporaryDirectory(prefix="shadowspill-demo-checkpoint-") as directory:
            copied = checkpoint.download(Path(directory) / "step-1").result(60)
            assert (
                decode((copied / "manifest.json").read_bytes()) == checkpoint_manifest
            )
            for row in checkpoint_manifest.ranks:
                archive = copied / "ranks" / f"rank-{row.rank:05d}.ssp"
                assert archive.stat().st_size == row.archive.size_bytes
                with archive.open("rb") as stream:
                    assert (
                        hashlib.file_digest(stream, "sha256").hexdigest()
                        == row.archive.sha256
                    )
        readonly = client.get_training_client(training.run_id, read_only=True).result(
            10
        )
        assert readonly.requests.get(step.request_id).result(10).record == result.record
        assert readonly.plans.summary().result(10) == summary[0]
        assert (
            readonly.requests.get(checkpoint_future.request_id).result(10).checkpoint
            == checkpoint
        )
        assert (
            readonly.requests.get(restore.request_id).result(10).record
            == restored.record
        )
        receipts.append(
            {
                "mode": mode,
                "run_id": training.run_id,
                "completed_updates": closed.completed_updates,
                "plans": saved.reference.to_dict(),
                "summary_pages": len(summary),
                "summary_rows": len(summary_rows),
                "summary_plan_count": page.plan_count,
                "checkpoint": checkpoint.reference.to_dict(),
                "checkpoint_rank_bytes": sum(
                    row.archive.size_bytes for row in checkpoint_manifest.ranks
                ),
                "checkpoint_save_seconds": save_seconds,
                "checkpoint_weights": checkpoint_manifest.info.weights,
                "checkpoint_download_after_close": True,
                "restore_request_id": restore.request_id,
                "restore_seconds": restore_seconds,
                "restored_updates": restored.completed_updates,
                "restored_history_id": restored.history_id,
                "post_restore_update_oracle": True,
            }
        )
    return receipts


raw = sys.stdin.buffer.read((1 << 20) + 1)
assert len(raw) <= 1 << 20
options = json.loads(raw)
memory = BufferBudget(64 << 20, 8)
connection = (
    {"ssh": SSH(options["ssh_host"], options["port"])}
    if options.get("ssh_host")
    else {"base_url": f"http://127.0.0.1:{options['port']}"}
)
with ServiceClient(
    **connection, token=options["token"], session_id="factory-session", timeout=30
) as client:
    receipts = qualify(
        client, ResolvedTraining.from_dict(options["config"]), memory, options["oracle"]
    )
assert client._memory.used == client._work.pending.used == (0, 0)
assert memory.used == (0, 0)
assert not any(name in sys.modules for name in ("torch", "triton", "shadowspill"))
print(
    json.dumps(
        {
            "client_host": socket.gethostname(),
            "public_training_client": True,
            "python": sys.version,
            "numpy": np.__version__,
            "runs": receipts,
            "all_client_claims_released": True,
        }
    )
)
