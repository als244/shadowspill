"""Client-only Llama training checks; eager reference values arrive as data."""

import importlib.abc
import json
import socket
import sys
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from time import monotonic

import numpy as np

from shadowspill_client import (
    SSH,
    BatchConfig,
    BuildCandidates,
    Datum,
    Memory,
    Model,
    Objective,
    Optimizer,
    PlanningOptions,
    Preparation,
    ProfilingOptions,
    ServiceClient,
)
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.payload_io import read_payload


def exported(training, trained, initial):
    handle = training.export_weights().result(120)
    with TemporaryDirectory(prefix="shadowspill-llama-weights-") as directory:
        root = handle.download(Path(directory) / "weights").result(60)
        with tarfile.open(root / "ranks/rank-00000.ssp") as archive:
            manifest = json.loads(archive.extractfile("manifest.json").read())
            arrays = {}
            endian = "<" if manifest["byteorder"] == "little" else ">"
            for row in manifest["tensors"]:
                dtype = row["dtype"].removeprefix("torch.")
                value = np.frombuffer(
                    archive.extractfile(row["member"]).read(),
                    dtype=np.dtype(
                        "uint16" if dtype == "bfloat16" else dtype
                    ).newbyteorder(endian),
                )
                if dtype == "bfloat16":
                    value = (value.astype(np.uint32) << 16).view(np.float32)
                arrays[row["name"]] = value
            values, masters = {}, set()
            for row in manifest["model_tensors"]:
                view, identity = row["view"], row["tensor"]["identity"]
                assert view["shape"] == view["local_shape"] and not any(view["origin"])
                if identity["role"] == "master":
                    assert row["tensor"]["geometry"]["dtype"] == "torch.float32"
                    masters.add(identity["name"])
                if row["archive"] in manifest["omitted_parameters"]:
                    assert manifest["frozen_state_id"] is not None
                    assert identity["name"] not in trained
                    values[identity["name"]] = np.asarray(
                        initial[identity["name"]], dtype=np.float32
                    )
                    continue
                array = arrays[
                    manifest["derived_tensors"].get(row["archive"], row["archive"])
                ]
                values[identity["name"]] = np.ndarray(
                    tuple(view["shape"]),
                    dtype=array.dtype,
                    buffer=array,
                    offset=(view["storage_offset"] - view["region_offset"])
                    * array.itemsize,
                    strides=tuple(s * array.itemsize for s in view["stride"]),
                ).copy()
            assert masters == set(trained), (masters, trained)
            return values


def logprobs(training, records, budget):
    values = [
        Datum(row["inputs"], {"target_tokens": row["loss_inputs"]["target_tokens"]})
        for row in records
    ]
    result = training.forward(values, outputs="logprobs").result(120)
    assert result.loss is None
    handle = result.outputs["logprobs"]
    with TemporaryDirectory(prefix="shadowspill-llama-logprobs-") as directory:
        path = handle.download(Path(directory) / "output.ssp").result(60)
        with (
            path.open("rb") as stream,
            read_payload(stream, handle.reference, budget) as decoded,
        ):
            fields = decoded.datums[0].inputs
            return [fields[str(index)].array().copy() for index in range(len(records))]


def check(training, options, index, budget):
    expected = options["oracle"][index]
    initial = options["oracle"][0]["weights"]
    actual = exported(training, options["trainable"], initial)
    for name, value in expected["weights"].items():
        value = np.asarray(value, dtype=np.float32)
        if index == 0 or name not in options["trainable"]:
            np.testing.assert_array_equal(actual[name], value)
        else:
            # The independent eager BF16 reference rounds weight VJPs to BF16;
            # capture retains FP32 gradients. Compare the update itself so large
            # unchanged weights cannot hide a missing or incorrectly scaled update.
            baseline = np.asarray(initial[name], dtype=np.float32)
            np.testing.assert_allclose(
                actual[name] - baseline,
                value - baseline,
                rtol=0.03,
                atol=1e-5,
                err_msg=name,
            )
    check_logprobs(training, options, index, budget)
    return actual


def check_logprobs(training, options, index, budget):
    for actual_row, expected_row in zip(
        logprobs(training, options["records"], budget),
        options["oracle"][index]["logprobs"],
        strict=True,
    ):
        np.testing.assert_allclose(actual_row, expected_row, rtol=0.003, atol=0.008)


def close(client, training):
    training.close().result(120)
    deadline = monotonic() + 30
    while True:
        info = client._transport.request(
            "GET", f"/v1/runs/{training.session_id}/{training.run_id}"
        ).value
        if info.state == "closed":
            return
        assert info.state == "ready" and monotonic() < deadline, info
        Event().wait(0.02)


def qualify(client, options, budget):
    preview = client.describe_training(
        model=Model.from_dict(options["model"]),
        objective=Objective.from_dict(options["objective"]),
        optimizer=Optimizer(
            "mlops/adamw@1",
            defaults=options["optim"],
            precision={
                "gradient_dtype": "float32",
                "opt_state_dtype": "float32",
                "master_dtype": "float32",
            },
        ),
        batch=BatchConfig(
            limits={"token_capacity": 8, "max_sequence_length": 4, "max_sequences": 4}
        ),
        memory=Memory(2, 0.125),
        operations={
            "step": ["metrics"],
            "forward_backward": ["metrics"],
            "forward": ["logprobs"],
        },
        prepare=Preparation(
            candidates=BuildCandidates(microbatch_sizes=[4, 8]),
            planning=PlanningOptions(microbatch_counts=[1, 2]),
            profiling=ProfilingOptions(
                conditioning_seconds=0.001, measurement_seconds=0.002
            ),
        ),
    ).result(30)
    training = client.create_training_client(config=preview.config).result(480)
    plans = training.plans
    initial = training.save_checkpoint(name="initial").result(120)
    records = [Datum(**row) for row in options["records"]]
    observations, whole = [], None
    for split in (False, True):
        if split:
            training.load_checkpoint(initial.id).result(120)
        check(training, options, 0, budget)
        if split:
            for part, subset in enumerate((records[:1], records[1:])):
                result = training.forward_backward(
                    subset, loss_params=options["controls"]
                ).result(120)
                np.testing.assert_allclose(
                    result.loss, options["loss_parts"][part], rtol=0.003, atol=0.008
                )
                observations.append(result.record.to_dict())
                # Export/checkpoint is intentionally forbidden while gradients
                # are pending. Read-only forward is the isolation check here.
                check_logprobs(training, options, 0, budget)
            result = training.optim_step(grad_scale=0.25).result(120)
        else:
            result = training.step(
                records, loss_params=options["controls"], grad_scale=0.25
            ).result(120)
            np.testing.assert_allclose(
                result.loss, sum(options["loss_parts"]), rtol=0.003, atol=0.008
            )
        assert result.outcome == "completed" and result.completed_updates == 1
        observations.append(result.record.to_dict())
        actual = check(training, options, 1, budget)
        if whole is None:
            whole = actual
        else:
            for name in options["trainable"]:
                np.testing.assert_allclose(
                    actual[name], whole[name], rtol=1e-4, atol=3e-6, err_msg=name
                )
    checkpoint = training.save_checkpoint(name="after-one").result(120)
    masked = [
        Datum(
            row.inputs,
            {**row.loss_inputs, options["mask"]: [False] * len(row.inputs["tokens"])},
        )
        for row in records
    ]
    result = training.step(masked, loss_params=options["controls"]).result(120)
    assert result.outcome == "skipped" and result.completed_updates == 1
    check(training, options, 1, budget)
    result = training.step(records, loss_params=options["controls"]).result(120)
    observations.append(result.record.to_dict())
    final = check(training, options, 2, budget)
    close(client, training)
    resumed = client.create_training_client(
        plans=plans, checkpoint=checkpoint, wait_for_resources=True
    ).result(240)
    check(resumed, options, 1, budget)
    result = resumed.step(records, loss_params=options["controls"]).result(120)
    assert result.completed_updates == 2
    observations.append(result.record.to_dict())
    actual = check(resumed, options, 2, budget)
    for name in options["trainable"]:
        np.testing.assert_allclose(
            actual[name], final[name], rtol=1e-4, atol=3e-6, err_msg=name
        )
    close(client, resumed)
    return {
        "runs": [training.run_id, resumed.run_id],
        "observations": observations,
        "whole_split_equivalence": True,
        "evaluation_isolation": True,
        "adamw_resume": True,
        "separate_fp32_masters": True,
    }


def main():
    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self, name, *args):
            if name.split(".")[0] in {
                "torch",
                "triton",
                "shadowspill",
                "shadowspill_workloads",
                "mlops",
            }:
                raise AssertionError("framework or engine import in client: " + name)

    sys.meta_path.insert(0, Block())
    raw = sys.stdin.buffer.read((4 << 20) + 1)
    assert len(raw) <= 4 << 20
    options = json.loads(raw)
    connection = (
        {"ssh": SSH(options["ssh_host"], options["port"])}
        if options.get("ssh_host")
        else {"base_url": f"http://127.0.0.1:{options['port']}"}
    )
    budget = BufferBudget(64 << 20, 8)
    with ServiceClient(
        **connection, token=options["token"], session_id="llama", timeout=30
    ) as client:
        receipt = qualify(client, options, budget)
    assert budget.used == client._memory.used == client._work.pending.used == (0, 0)
    receipt.update(client_host=socket.gethostname(), all_client_claims_released=True)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
