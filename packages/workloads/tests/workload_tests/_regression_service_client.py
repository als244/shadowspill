"""NumPy-only supplied-model qualification over the ordinary public client."""

import copy
import importlib.abc
import json
import socket
import sys
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory

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
    Weights,
)
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.payload_io import read_payload

INPUTS = np.linspace(-0.7, 0.9, 5 * 16, dtype=np.float32).reshape(5, 16)
TARGETS = np.linspace(0.8, -0.6, 5 * 8, dtype=np.float32).reshape(5, 8)
WEIGHTS = np.array([1, -0.25, 0, 0.75, 0.5], dtype=np.float32) / 40
VALID = np.array([True, True, True, False, True])
DEFAULTS = {
    "lr": 0.001,
    "beta1": 0.9,
    "beta2": 0.999,
    "eps": 1e-8,
    "weight_decay": 0.01,
}


def bfloat16(value):
    bits = np.asarray(value, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000).view(np.float32)


class Oracle:
    """Explicit SiLU derivatives and AdamW equations, independent of the engine."""

    def __init__(self, initial, gradient_dtype, master_dtype=None):
        self.weights = {
            name: np.asarray(value, dtype=np.float32).copy()
            for name, value in initial.items()
        }
        self.moments = {
            name: [np.zeros_like(value), np.zeros_like(value)]
            for name, value in self.weights.items()
        }
        self.gradient_dtype = gradient_dtype
        self.master_dtype = master_dtype
        self.steps = 0

    def forward_backward(
        self, selection=slice(None), *, inputs=INPUTS, targets=TARGETS, weights=None
    ):
        x, target = inputs[selection], targets[selection]
        weights = (WEIGHTS * VALID if weights is None else weights)[selection]
        w = self.weights
        z = x @ w["network.0.weight"].T + w["network.0.bias"]
        sigmoid = 1 / (1 + np.exp(-z))
        activation = z * sigmoid
        prediction = activation @ w["network.2.weight"].T + w["network.2.bias"]
        error = prediction.astype(np.float64) - target
        derivative = 2 * error * weights[:, None]
        hidden = (derivative @ w["network.2.weight"]) * (
            sigmoid + z * sigmoid * (1 - sigmoid)
        )
        return (
            prediction,
            (error**2 * weights[:, None]).sum(),
            {
                "network.2.weight": derivative.T @ activation,
                "network.2.bias": derivative.sum(axis=0),
                "network.0.weight": hidden.T @ x,
                "network.0.bias": hidden.sum(axis=0),
            },
        )

    def update(self, gradients, optim=None, scale=1):
        controls = {
            name: float(np.float32(value))
            for name, value in {**DEFAULTS, **(optim or {})}.items()
        }
        rate, beta1, beta2, eps, decay = (controls[name] for name in DEFAULTS)
        if rate == 0:
            return  # The existing MLOps contract preserves moments and counters.
        self.steps += 1
        for name, value in self.weights.items():
            gradient = np.asarray(gradients[name] * scale, dtype=np.float32)
            if self.gradient_dtype == "bfloat16":
                gradient = bfloat16(gradient)
            first, second = self.moments[name]
            first[:] = beta1 * first + (1 - beta1) * gradient
            second[:] = beta2 * second + (1 - beta2) * gradient**2
            value[:] = value * (1 - rate * decay) - rate * (
                first / (1 - beta1**self.steps)
            ) / (np.sqrt(second / (1 - beta2**self.steps)) + eps)


def records(*, targets=True, masked=False):
    return [
        Datum(
            {"value": value},
            {
                "targets": TARGETS[index],
                "weights": float(WEIGHTS[index]),
                "valid": bool(VALID[index]) and not masked,
            }
            if targets
            else {},
        )
        for index, value in enumerate(INPUTS)
    ]


def prediction(training, memory):
    result = training.forward(records(targets=False)).result(120)
    assert result.loss is None
    handle = result.outputs["prediction"]
    with TemporaryDirectory(prefix="shadowspill-regression-prediction-") as directory:
        path = handle.download(Path(directory) / "output.ssp").result(30)
        with (
            path.open("rb") as stream,
            read_payload(stream, handle.reference, memory) as decoded,
        ):
            rows = decoded.datums[0].inputs
            return np.stack([rows[str(i)].array() for i in range(len(INPUTS))])


def weights(training):
    handle = training.export_weights().result(120)
    with TemporaryDirectory(prefix="shadowspill-regression-weights-") as directory:
        root = handle.download(Path(directory) / "weights").result(60)
        with tarfile.open(root / "ranks/rank-00000.ssp") as archive:
            manifest = json.loads(archive.extractfile("manifest.json").read())
            tensors = {
                row["name"]: np.frombuffer(
                    archive.extractfile(row["member"]).read(),
                    dtype=np.dtype(row["dtype"].removeprefix("torch.")).newbyteorder(
                        "<" if manifest["byteorder"] == "little" else ">"
                    ),
                )
                for row in manifest["tensors"]
            }
            result = {}
            for row in manifest["model_tensors"]:
                view, identity = row["view"], row["tensor"]["identity"]
                # This model computes in FP32. Requesting FP32 optimizer masters
                # reuses those parameters; it must not allocate redundant copies.
                assert identity["role"] in ("parameter", "buffer")
                assert view["shape"] == view["local_shape"] and not any(view["origin"])
                source = manifest["derived_tensors"].get(row["archive"], row["archive"])
                value = tensors[source]
                result[identity["name"]] = np.ndarray(
                    tuple(view["shape"]),
                    dtype=value.dtype,
                    buffer=value,
                    offset=(view["storage_offset"] - view["region_offset"])
                    * value.itemsize,
                    strides=tuple(stride * value.itemsize for stride in view["stride"]),
                ).copy()
            return result


def check(training, oracle, memory, *, counter=None):
    actual = weights(training)
    assert actual.keys() == oracle.weights.keys() | {"updates"}
    for name, expected in oracle.weights.items():
        np.testing.assert_allclose(actual[name], expected, rtol=3e-4, atol=3e-6)
    if counter is not None:
        assert actual["updates"].item() == counter
    np.testing.assert_allclose(
        prediction(training, memory), oracle.forward_backward()[0], rtol=3e-4, atol=3e-6
    )
    return actual


def closed(client, training):
    training.close().result(120)
    reply = client._transport.request(
        "GET", f"/v1/runs/regression/{training.run_id}"
    )
    assert reply.status == 200 and reply.value.state == "closed"


def qualify(client, initial, dtype, memory, master_dtype=None):
    # Setup is separate from the loop; describe never constructs a model/worker.
    preview = client.describe_training(
        model=Model("shadowspill/regression@1", initialize=Weights.random(seed=0)),
        objective=Objective("shadowspill/weighted_squared_error@1"),
        optimizer=Optimizer(
            "mlops/adamw@1",
            precision={
                "gradient_dtype": dtype,
                "opt_state_dtype": "float32",
                "master_dtype": master_dtype,
            },
        ),
        batch=BatchConfig(limits={"max_examples": 8}),
        memory=Memory(device_gib=2, spill_gib=0.0625),
        operations={
            "step": ["metrics"],
            "forward_backward": ["metrics"],
            "forward": ["predictions"],
        },
        prepare=Preparation(
            candidates=BuildCandidates(microbatch_sizes=[4]),
            planning=PlanningOptions(microbatch_counts=[1, 2]),
            profiling=ProfilingOptions(
                conditioning_seconds=0.001, measurement_seconds=0.002
            ),
        ),
    ).result(30)
    training = client.create_training_client(config=preview.config).result(240)
    plans, observations = training.plans, []
    initial_checkpoint = training.save_checkpoint(name="initial").result(120)
    first = None
    for split in (False, True):
        if split:
            training.load_checkpoint(initial_checkpoint.id).result(120)
        oracle = Oracle(initial, dtype, master_dtype)
        check(training, oracle, memory, counter=0)
        optim = {
            "lr": 0.003,
            "beta1": 0.8,
            "beta2": 0.95,
            "eps": 1e-6,
            "weight_decay": 0.02,
        }
        if split:
            for start, stop in ((0, 2), (2, 5)):
                result = training.forward_backward(records()[start:stop]).result(120)
                np.testing.assert_allclose(
                    result.loss,
                    oracle.forward_backward(slice(start, stop))[1],
                    rtol=2e-5,
                    atol=1e-6,
                )
                np.testing.assert_allclose(
                    prediction(training, memory),
                    oracle.forward_backward()[0],
                    rtol=2e-5,
                    atol=1e-6,
                )
                observations.append(result.record.to_dict())
            result = training.optim_step(optim=optim, grad_scale=0.5).result(120)
        else:
            result = training.step(records(), optim=optim, grad_scale=0.5).result(120)
            np.testing.assert_allclose(
                result.loss, oracle.forward_backward()[1], rtol=2e-5, atol=1e-6
            )
        observations.append(result.record.to_dict())
        assert result.completed_updates == 1 and result.outcome == "completed"
        oracle.update(oracle.forward_backward()[2], optim, 0.5)
        actual = check(training, oracle, memory, counter=2)
        if first is None:
            first = actual
        else:
            for name in actual:
                np.testing.assert_allclose(
                    actual[name], first[name], rtol=2e-5, atol=1e-6
                )

    checkpoint = training.save_checkpoint(name="after-one").result(120)
    checkpoint_oracle = copy.deepcopy(oracle)
    result = training.step(records(masked=True)).result(120)
    assert result.outcome == "skipped" and result.completed_updates == 1
    check(training, oracle, memory, counter=2)
    observations.append(result.record.to_dict())
    for optim in ({"lr": 0.0}, None):
        result = training.step(records(), optim=optim).result(120)
        np.testing.assert_allclose(
            result.loss, oracle.forward_backward()[1], rtol=2e-5, atol=1e-6
        )
        oracle.update(oracle.forward_backward()[2], optim)
        check(training, oracle, memory)
        observations.append(result.record.to_dict())
    final = weights(training)
    closed(client, training)
    resumed = client.create_training_client(
        plans=plans, checkpoint=checkpoint, wait_for_resources=True
    ).result(240)
    oracle = checkpoint_oracle
    check(resumed, oracle, memory, counter=2)
    result = resumed.step(records()).result(120)
    oracle.update(oracle.forward_backward()[2])
    actual = check(resumed, oracle, memory, counter=4)
    for name in oracle.weights:
        np.testing.assert_allclose(actual[name], final[name], rtol=2e-5, atol=1e-6)
    observations.append(result.record.to_dict())
    closed(client, resumed)
    return {
        "runs": [training.run_id, resumed.run_id],
        "observations": observations,
        "gradient_dtype": dtype,
        "master_dtype": master_dtype,
        "whole_split_equivalence": True,
        "evaluation_isolation": True,
        "adamw_resume": True,
        "zero_lr_preserves_moments": True,
    }


def main():
    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, *args):
            if fullname.split(".")[0] in {
                "torch",
                "triton",
                "shadowspill",
                "shadowspill_workloads",
                "mlops",
            }:
                raise AssertionError("framework/engine import in client: " + fullname)

    sys.meta_path.insert(0, Block())
    raw = sys.stdin.buffer.read((1 << 20) + 1)
    assert len(raw) <= 1 << 20
    options = json.loads(raw)
    connection = (
        {"ssh": SSH(options["ssh_host"], options["port"])}
        if options.get("ssh_host")
        else {"base_url": f"http://127.0.0.1:{options['port']}"}
    )
    memory = BufferBudget(64 << 20, 8)
    with ServiceClient(
        **connection, token=options["token"], session_id="regression", timeout=30
    ) as client:
        receipt = qualify(
            client,
            options["initial"],
            options["gradient_dtype"],
            memory,
            options["master_dtype"],
        )
    assert memory.used == client._memory.used == client._work.pending.used == (0, 0)
    receipt.update(
        client_host=socket.gethostname(),
        numpy=np.__version__,
        all_client_claims_released=True,
    )
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
