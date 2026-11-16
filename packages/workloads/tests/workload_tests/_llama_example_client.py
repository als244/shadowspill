"""Exercise the actual Llama CLI and resume with a NumPy-only demo client."""

import importlib.abc
import json
import os
import socket
import sys
import tarfile
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split(".")[0] in {
            "torch",
            "triton",
            "mlops",
            "shadowspill",
            "shadowspill_workloads",
        }:
            raise AssertionError("client imported " + fullname)


sys.meta_path.insert(0, Block())
# A copied example tree lives beside this script on Tubingen; a checkout does
# on Chicago. Neither route installs the engine or supplied model package.
here = Path(__file__).resolve().parent
application = here if (here / "examples").is_dir() else here.parents[3]
sys.path.insert(0, str(application))
from examples.train_llama3 import main  # noqa: E402
from shadowspill_client import SSH, ServiceClient  # noqa: E402
from shadowspill_client.protocol.requests import ArtifactRef  # noqa: E402


def completed_run(service, directory):
    info = json.loads((directory / "run.json").read_text())
    route = f"/v1/runs/{info['session_id']}/{info['run_id']}"
    reply = service._transport.request("GET", route)
    assert reply.status == 200 and reply.value.state == "closed"
    return info["run_id"]


def final_checkpoint(service, directory, destination):
    record = json.loads((directory / "checkpoints/step-00000003.json").read_text())
    checkpoint = service.artifact(ArtifactRef.from_dict(record["checkpoint"]))
    info = checkpoint.manifest().result().info
    assert info.completed_updates == info.metadata["next_update"] == 3
    checkpoint.download(destination).result()
    with tarfile.open(destination / "ranks/rank-00000.ssp") as archive:
        manifest = json.load(archive.extractfile("manifest.json"))
    return {row["name"]: row["blake3"] for row in manifest["tensors"]}


def qualify(options):
    os.environ["SHADOWSPILL_TOKEN"] = options["token"]
    connection = (
        {"ssh": SSH(options["ssh_host"], options["port"])}
        if options.get("ssh_host")
        else {"base_url": f"http://127.0.0.1:{options['port']}"}
    )
    flags = (
        ["--ssh", options["ssh_host"], "--server-port", str(options["port"])]
        if options.get("ssh_host")
        else ["--server", connection["base_url"]]
    )
    with (
        TemporaryDirectory(prefix="shadowspill-llama-example-") as temporary,
        ServiceClient(**connection, token=options["token"]) as service,
    ):
        root = Path(temporary)
        data = root / "data"
        data.mkdir()
        values = np.arange(4096, dtype=np.uint32) % 32
        values.tofile(data / "train.bin")
        ((values + 7) % 32).tofile(data / "val.bin")
        (data / "meta.json").write_text(json.dumps({"vocab_size": 512}))
        first, resumed = root / "fresh", root / "resumed"
        with redirect_stdout(sys.stderr):
            assert (
                main(
                    [
                        *flags,
                        "--data",
                        str(data),
                        "--run-dir",
                        str(first),
                        "--model",
                        "llama3-tiny",
                        "--sequence-length",
                        "4",
                        "--sequences-per-step",
                        "3",
                        "--sequences-per-example",
                        "1",
                        "--backward-pieces",
                        "2",
                        "--steps",
                        "3",
                        "--checkpoint-every",
                        "1",
                        "--learning-rate",
                        "0.001",
                        "--execution-gib",
                        "2",
                        "--spill-gib",
                        "0.125",
                        "--warmup-seconds",
                        "0.001",
                        "--measure-seconds",
                        "0.002",
                    ]
                )
                == 0
            )
        run = completed_run(service, first)
        with redirect_stdout(sys.stderr):
            assert (
                main(
                    [
                        *flags,
                        "--run-dir",
                        str(resumed),
                        "--resume",
                        str(first / "checkpoints/step-00000001.json"),
                    ]
                )
                == 0
            )
        restored = completed_run(service, resumed)
        metrics = [
            json.loads(line)
            for line in (first / "metrics.jsonl").read_text().splitlines()
        ]
        continuation = [
            json.loads(line)
            for line in (resumed / "metrics.jsonl").read_text().splitlines()
        ]
        assert [row["step"] for row in metrics] == [1, 2, 3]
        assert [row["step"] for row in continuation] == [2, 3]
        for name in ("loss", "lr"):
            assert [row[name] for row in metrics[1:]] == [
                row[name] for row in continuation
            ]
        assert all(np.isfinite(row["loss"]) for row in metrics)
        assert final_checkpoint(
            service, first, root / "first-checkpoint"
        ) == final_checkpoint(service, resumed, root / "resumed-checkpoint")
        for path in (first, resumed):
            assert (path / "training.html").stat().st_size > 100
        return {
            "client_host": socket.gethostname(),
            "runs": [run, restored],
            "metrics": metrics,
            "resumed_metrics": continuation,
            "exact_checkpoint_continuation": True,
            "actual_example": True,
        }


raw = sys.stdin.buffer.read((4 << 20) + 1)
assert len(raw) <= 4 << 20
print(json.dumps(qualify(json.loads(raw))))
