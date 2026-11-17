"""Exercise the actual Qwen CLI using only NumPy and the lightweight SDK."""

import importlib.abc
import json
import os
import socket
import sys
import tarfile
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from time import monotonic

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
application = next(
    parent for parent in (here, *here.parents) if (parent / "examples").is_dir()
)
sys.path.insert(0, str(application))
from examples import train_qwen3_distributed, train_qwen35_lora  # noqa: E402
from shadowspill_client import SSH, ServiceClient  # noqa: E402
from shadowspill_client.protocol.requests import ArtifactRef  # noqa: E402


def completed(service, directory):
    info = json.loads((directory / "run.json").read_text())
    deadline = monotonic() + 30
    while True:
        response = service._transport.request(
            "GET", f"/v1/runs/{info['session_id']}/{info['run_id']}"
        )
        assert response.status == 200
        if response.value.state == "closed":
            return info["run_id"]
        assert response.value.state == "ready" and monotonic() < deadline
        Event().wait(0.02)


def checkpoint(service, directory, destination, trained, *, lora):
    saved = json.loads((directory / "checkpoints/step-00000003.json").read_text())
    handle = service.artifact(ArtifactRef.from_dict(saved["checkpoint"]))
    info = handle.manifest().result().info
    assert info.completed_updates == info.metadata["next_update"] == 3
    handle.download(destination).result()
    with tarfile.open(destination / "ranks/rank-00000.ssp") as archive:
        manifest = json.load(archive.extractfile("manifest.json"))
        assert (manifest["frozen_state_id"] is not None) == lora
        assert ("embed.weight" in manifest["omitted_parameters"]) == lora
        hashes = {row["name"]: row["blake3"] for row in manifest["tensors"]}
        # This fixture uses one worker and untied FP32 weights. Native checkpoint
        # tensor descriptors give their complete geometry without engine imports.
        assert manifest["logical_rank"] == 0 and not manifest["gathered_tensors"]
        tensors = {row["name"]: row for row in manifest["tensors"]}
        weights = {}
        for name, expected in trained.items():
            assert name not in manifest["omitted_parameters"]
            row = tensors[manifest["derived_tensors"].get(name, name)]
            dtype = np.dtype(row["dtype"].removeprefix("torch.")).newbyteorder(
                "<" if manifest["byteorder"] == "little" else ">"
            )
            array = np.frombuffer(
                archive.extractfile(row["member"]).read(), dtype=dtype
            )
            assert array.dtype == np.float32 and row.get("region_offset", 0) == 0
            assert array.size == row["storage_elements"]
            assert tuple(row["shape"]) == np.shape(expected)
            weights[name] = np.ndarray(
                tuple(row["shape"]),
                dtype=array.dtype,
                buffer=array,
                offset=row["storage_offset"] * array.itemsize,
                strides=tuple(n * array.itemsize for n in row["stride"]),
            ).copy()
    return hashes, weights


def qualify(options):
    main = train_qwen35_lora.main if options["lora"] else train_qwen3_distributed.main
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
        TemporaryDirectory(prefix="shadowspill-qwen-example-") as temporary,
        ServiceClient(**connection, token=options["token"]) as service,
    ):
        root = Path(temporary)
        data = root / "data"
        data.mkdir()
        tokens = np.arange(97, dtype=np.uint32) % 16
        tokens.tofile(data / "train.bin")
        ((tokens + 1) % 16).tofile(data / "val.bin")
        (data / "meta.json").write_text(json.dumps({"vocab_size": 16}))
        first, resumed = root / "fresh", root / "resumed"
        with redirect_stdout(sys.stderr):
            assert (
                main(
                    [
                        *flags,
                        "--data",
                        str(data),
                        "--checkpoint",
                        options["checkpoint"],
                        "--run-dir",
                        str(first),
                        "--sequence-length",
                        "4",
                        "--ranks",
                        "1",
                        "--sequences-per-step",
                        "2",
                        "--microbatch-sizes",
                        "2",
                        "--steps",
                        "3",
                        "--schedule-steps",
                        "3",
                        *(
                            ["--lora-rank", "2", "--lora-alpha", "2"]
                            if options["lora"]
                            else []
                        ),
                        "--seed",
                        "45",
                        "--compute-dtype",
                        "float32",
                        "--learning-rate",
                        "0.001",
                        "--backward-pieces",
                        str(options["pieces"]),
                        "--eval-every",
                        "1",
                        "--eval-updates",
                        "1",
                        "--checkpoint-every",
                        "1",
                        "--execution-gib",
                        "6",
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
        run = completed(service, first)
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
        restored = completed(service, resumed)
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
        np.testing.assert_allclose(
            [row["loss"] for row in metrics], options["losses"], rtol=1e-5, atol=2e-5
        )
        np.testing.assert_allclose(
            [row["lr"] for row in metrics], options["rates"], rtol=1e-7, atol=0
        )
        evaluated = [
            json.loads(line)
            for line in (first / "evaluation.jsonl").read_text().splitlines()
        ]
        np.testing.assert_allclose(
            [row["validation_loss"] for row in evaluated],
            options["evaluations"],
            rtol=1e-5,
            atol=2e-5,
        )
        hashes, actual = checkpoint(
            service,
            first,
            root / "first-checkpoint",
            options["initial"],
            lora=options["lora"],
        )
        resumed_hashes, _ = checkpoint(
            service,
            resumed,
            root / "resumed-checkpoint",
            options["initial"],
            lora=options["lora"],
        )
        assert hashes == resumed_hashes
        for name, value in actual.items():
            initial = np.asarray(options["initial"][name], np.float32)
            np.testing.assert_allclose(
                value - initial,
                np.asarray(options["final"][name], np.float32) - initial,
                rtol=0.003,
                atol=2e-6,
                err_msg=name,
            )
        for directory in (first, resumed):
            assert (directory / "training.html").stat().st_size > 100
        result = {
            "client_host": socket.gethostname(),
            "runs": [run, restored],
            "pieces": options["pieces"],
            "family": options["family"],
            "metrics": metrics,
            "resumed_metrics": continuation,
            "actual_example": True,
            "independent_loss_and_weights": True,
            "exact_checkpoint_continuation": True,
        }
    assert service._memory.used == service._work.pending.used == (0, 0)
    result["all_client_claims_released"] = True
    return result


raw = sys.stdin.buffer.read((8 << 20) + 1)
assert len(raw) <= 8 << 20
print(json.dumps(qualify(json.loads(raw))))
