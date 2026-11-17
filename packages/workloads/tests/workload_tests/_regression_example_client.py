"""Run the actual regression CLI with a NumPy oracle on a lightweight client."""

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
here = Path(__file__).resolve().parent
application = here if (here / "examples").is_dir() else here.parents[3]
sys.path.insert(0, str(application))
from _regression_service_client import Oracle  # noqa: E402

from examples.regression_training import main  # noqa: E402
from examples.training.regression.data import SyntheticBatches  # noqa: E402
from shadowspill_client import SSH, ServiceClient  # noqa: E402
from shadowspill_client.protocol.requests import ArtifactRef  # noqa: E402


def completed_run(service, directory):
    info = json.loads((directory / "run.json").read_text())
    route = f"/v1/runs/{info['session_id']}/{info['run_id']}"
    reply = service._transport.request("GET", route)
    assert reply.status == 200 and reply.value.state == "closed"
    return info["run_id"]


def final_checkpoint(service, directory, destination, initial, ranks):
    record = json.loads((directory / "checkpoints/step-00000003.json").read_text())
    checkpoint = service.artifact(ArtifactRef.from_dict(record["checkpoint"]))
    info = checkpoint.manifest().result().info
    assert info.completed_updates == info.metadata["next_update"] == 3
    checkpoint.download(destination).result()
    values = {
        name: np.zeros(np.size(value), np.float32) for name, value in initial.items()
    }
    covered = {name: np.zeros(value.shape, bool) for name, value in values.items()}
    hashes = {}
    for rank in range(ranks):
        with tarfile.open(destination / f"ranks/rank-{rank:05d}.ssp") as archive:
            manifest = json.load(archive.extractfile("manifest.json"))
            assert not manifest["derived_tensors"] and not manifest["gathered_tensors"]
            for row in manifest["tensors"]:
                hashes[rank, row["name"]] = row["blake3"]
                if row["name"] not in initial:
                    continue
                assert row["dtype"] == "torch.float32" and row["storage_offset"] == 0
                value = np.frombuffer(
                    archive.extractfile(row["member"]).read(),
                    dtype="<f4" if manifest["byteorder"] == "little" else ">f4",
                )
                start = row.get("region_offset", 0)
                end = min(start + value.size, values[row["name"]].size)
                owned = value[: end - start]
                destination_values = values[row["name"]][start:end]
                already = covered[row["name"]][start:end]
                np.testing.assert_array_equal(
                    destination_values[already], owned[already]
                )
                destination_values[:] = owned
                already[:] = True
    assert all(part.all() for part in covered.values())
    return hashes, {
        name: value.reshape(np.shape(initial[name])) for name, value in values.items()
    }


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
        TemporaryDirectory(prefix="shadowspill-regression-example-") as temporary,
        ServiceClient(**connection, token=options["token"]) as service,
    ):
        root = Path(temporary)
        first, resumed = root / "fresh", root / "resumed"
        with redirect_stdout(sys.stderr):
            assert (
                main(
                    [
                        *flags,
                        "--run-dir",
                        str(first),
                        "--ranks",
                        str(options["ranks"]),
                        "--steps",
                        "3",
                        "--checkpoint-every",
                        "1",
                        "--backward-pieces",
                        str(options["pieces"]),
                        "--execution-gib",
                        "4" if options["ranks"] > 1 else "2",
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
        for row in metrics + continuation:
            assert 0 < row["execution_seconds"] <= row["client_elapsed_seconds"]
        assert [row["loss"] for row in metrics[1:]] == [
            row["loss"] for row in continuation
        ]
        initial = options["initial"]
        oracle = Oracle(initial, "float32")
        source = SyntheticBatches(ranks=options["ranks"])
        for index, row in enumerate(metrics):
            batch = source[index]
            _, expected, gradients = oracle.forward_backward(
                inputs=np.stack([datum.inputs["value"] for datum in batch]),
                targets=np.stack([datum.loss_inputs["targets"] for datum in batch]),
                weights=np.array(
                    [datum.loss_inputs["weights"] for datum in batch], np.float32
                ),
            )
            np.testing.assert_allclose(row["loss"], expected, rtol=2e-5, atol=1e-6)
            oracle.update(gradients)
        hashes, actual = final_checkpoint(
            service, first, root / "first-checkpoint", initial, options["ranks"]
        )
        resumed_hashes, _ = final_checkpoint(
            service, resumed, root / "resumed-checkpoint", initial, options["ranks"]
        )
        assert hashes == resumed_hashes
        for name, value in actual.items():
            np.testing.assert_allclose(
                value - np.array(initial[name], np.float32),
                oracle.weights[name] - np.array(initial[name], np.float32),
                rtol=3e-4,
                atol=3e-6,
            )
        for path in (first, resumed):
            assert (path / "training.html").stat().st_size > 100
        return {
            "client_host": socket.gethostname(),
            "runs": [run, restored],
            "ranks": options["ranks"],
            "pieces": options["pieces"],
            "metrics": metrics,
            "resumed_metrics": continuation,
            "exact_checkpoint_continuation": True,
            "actual_example": True,
            "independent_loss_and_weights": True,
        }


raw = sys.stdin.buffer.read((4 << 20) + 1)
assert len(raw) <= 4 << 20
print(json.dumps(qualify(json.loads(raw))))
