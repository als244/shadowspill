"""NumPy-only public-client training, independent gradients and stochastic replay."""

import importlib.abc
import io
import json
import socket
import sys
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from time import monotonic
from zipfile import ZipFile

import numpy as np

from shadowspill_client import (
    SSH,
    BuildCandidates,
    Datum,
    PlanningOptions,
    Preparation,
    ProfilingOptions,
    ServiceClient,
)
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.checkpoints import CheckpointHandle
from shadowspill_client.protocol.payload_io import read_payload
from shadowspill_client.protocol.training import ResolvedTraining

PIXELS = np.stack(
    [np.arange(4, dtype=np.float32) / 8 + index / 4 for index in range(3)]
)
WEIGHTS = np.array([1.0, -0.5, 0.0])


class Oracle:
    """Explicit tanh/softmax derivatives; no engine or autodiff dependency."""

    def __init__(self, initial):
        self.weights = {
            name: np.asarray(value, dtype=np.float32).copy()
            for name, value in initial.items()
        }
        self.gradients = {
            name: np.zeros(value.shape, dtype=np.float64)
            for name, value in self.weights.items()
        }

    def forward_backward(self, noise, temperature=1.0):
        weight = self.weights
        hidden = np.tanh(PIXELS @ weight["encoder.weight"].T + weight["encoder.bias"])
        activation = hidden * noise
        prediction = activation @ weight["head.weight"].T + weight["head.bias"]
        logits = prediction.astype(np.float64) / temperature
        logits -= logits.max(axis=1, keepdims=True)
        exponential = np.exp(logits)
        probabilities = exponential / exponential.sum(axis=1, keepdims=True)
        loss = -(np.log(probabilities[np.arange(3), np.arange(3)]) * WEIGHTS).sum()
        derivative = probabilities.copy()
        derivative[np.arange(3), np.arange(3)] -= 1
        derivative *= WEIGHTS[:, None] / temperature
        hidden_derivative = (
            (derivative @ weight["head.weight"]) * noise * (1 - hidden**2)
        )
        gradients = {
            "head.weight": derivative.T @ activation,
            "head.bias": derivative.sum(axis=0),
            "encoder.weight": hidden_derivative.T @ PIXELS,
            "encoder.bias": hidden_derivative.sum(axis=0),
        }
        return prediction, loss, gradients

    def check(self, outputs, result, *, temperature=1.0, backward=False):
        noise = outputs["noise"]
        assert noise.shape == (3, 8) and np.all((noise >= 0.5) & (noise < 1.5))
        prediction, loss, gradients = self.forward_backward(noise, temperature)
        np.testing.assert_allclose(
            outputs["prediction"], prediction, rtol=2e-4, atol=2e-5
        )
        if backward:
            np.testing.assert_allclose(result.loss, loss, rtol=2e-4, atol=2e-5)
            for name, value in gradients.items():
                self.gradients[name] += value
        return noise.copy()

    def update(self, rate, scale):
        for name, value in self.weights.items():
            value -= rate * scale * self.gradients[name]
            self.gradients[name].fill(0)


def records(*, targets=True, masked=False):
    return [
        Datum(
            {"pixels": row.reshape(1, 2, 2)},
            {"class_id": index, "weights": float(WEIGHTS[index]), "valid": not masked}
            if targets
            else {},
        )
        for index, row in enumerate(PIXELS)
    ]


def outputs(result, memory):
    values = {}
    with TemporaryDirectory(prefix="shadowspill-rng-output-") as directory:
        for name, handle in result.outputs.items():
            path = handle.download(Path(directory) / (name + ".ssp")).result(30)
            with (
                path.open("rb") as stream,
                read_payload(stream, handle.reference, memory) as decoded,
            ):
                rows = decoded.datums[0].inputs
                values[name] = np.stack(
                    [rows[str(index)].array() for index in range(3)]
                )
    return values


def closed(client, training):
    training.close().result(30)
    deadline = monotonic() + 30
    while True:
        info = client._transport.request(
            "GET", "/v1/runs/rng-session/" + training.run_id
        ).value
        if info.state == "closed":
            return
        assert monotonic() < deadline, info
        Event().wait(0.02)


def check_trace(result, run, *, expected, accumulated=False):
    assert not result.artifact_issues, result.artifact_issues
    if not expected:
        assert not result.traces
        return
    assert len(result.traces) == 1
    with TemporaryDirectory(prefix="shadowspill-trace-output-") as directory:
        path = result.traces[0].download(Path(directory) / "trace.zip").result(30)
        with ZipFile(path) as archive:
            assert set(archive.namelist()) == {
                "manifest.json",
                "step.json",
                "step.events.npz",
            }
            manifest = json.loads(archive.read("manifest.json"))
            assert manifest["request"] == result.identity.to_dict()
            assert manifest["operation"] == result.operation
            assert manifest["plans"] == run.plans.reference.to_dict()
            assert manifest["rank"] == 0 and manifest["history_id"] == result.history_id
            assert manifest["completed_updates_before"] == result.completed_updates - (
                result.operation in ("step", "optim_step")
                and result.outcome != "skipped"
            )
            assert manifest["accumulate_gradients"] == accumulated
            diagnostic = json.loads(archive.read("step.json"))
            assert diagnostic["tasks"] and diagnostic["timelines"]["compute"]
            assert diagnostic["summary"]["trace_complete"]
            assert not diagnostic["runtime"]["event_overflow"]
            assert not diagnostic["allocator"]["overflow"]
            with np.load(
                io.BytesIO(archive.read("step.events.npz")), allow_pickle=False
            ) as events:
                assert len(events["runtime.sequence"]) >= 2
                for name in ("runtime", "allocator"):
                    sequence = events[name + ".sequence"]
                    np.testing.assert_array_equal(sequence, np.arange(len(sequence)))


def qualify(client, config, initial, memory):
    training = client.create_training_client(config=config, name="stochastic").result(
        180
    )
    plans = training.plans
    oracle = Oracle(initial)
    first = training.step(
        records(),
        optim={"lr": 0.03},
        grad_scale=0.5,
        loss_params={"temperature": 2.0},
        runtime_trace=True,
    ).result(120)
    # The server fixture fails this optional artifact publication after the real
    # device update. Loss, model state and the update counter must still succeed.
    assert not first.traces and first.completed_updates == 1 and first.state_valid
    assert len(first.artifact_issues) == 1
    assert first.artifact_issues[0].code == "runtime_trace_publication"
    oracle.check(outputs(first, memory), first, temperature=2.0, backward=True)
    oracle.update(0.03, 0.5)
    evaluated = training.forward(
        records(targets=False), outputs="predictions", runtime_trace=True
    ).result(120)
    oracle.check(outputs(evaluated, memory), evaluated)
    check_trace(evaluated, training, expected=True)
    retained_trace = evaluated.traces[0]
    checkpoint = training.save_checkpoint(name="base", metadata={"cursor": 3}).result(
        120
    )
    initial_resume = {name: value.copy() for name, value in oracle.weights.items()}
    assert checkpoint.manifest().result(30).info.completed_updates == 1
    receipts = []

    def branch(run, split, interpose):
        reference = Oracle(initial_resume)
        observed = []

        def evaluate(masked=False):
            result = (
                run.forward_backward(records(masked=True), runtime_trace=interpose)
                if masked
                else run.forward(
                    records(targets=False),
                    outputs="predictions",
                    runtime_trace=interpose,
                )
            ).result(120)
            reference.check(outputs(result, memory), result)
            check_trace(result, run, expected=interpose)
            if masked:
                assert result.record.outcome == "skipped"

        if interpose:
            evaluate()
        if split:
            for index, temperature in enumerate((0.5, 1.0)):
                result = run.forward_backward(
                    records(),
                    loss_params={"temperature": temperature},
                    runtime_trace=interpose,
                ).result(120)
                check_trace(result, run, expected=interpose, accumulated=index > 0)
                observed.append(
                    reference.check(
                        outputs(result, memory),
                        result,
                        temperature=temperature,
                        backward=True,
                    )
                )
                if interpose and index == 0:
                    evaluate()
                    evaluate(masked=True)
            final = run.optim_step(
                optim={"lr": 0.02}, grad_scale=-0.5, runtime_trace=interpose
            ).result(120)
            check_trace(final, run, expected=interpose)
            reference.update(0.02, -0.5)
        else:
            result = run.step(
                records(),
                optim={"lr": 0.04},
                grad_scale=1.25,
                loss_params={"temperature": 1.5},
                runtime_trace=interpose,
            ).result(120)
            check_trace(result, run, expected=interpose)
            observed.append(
                reference.check(
                    outputs(result, memory), result, temperature=1.5, backward=True
                )
            )
            reference.update(0.04, 1.25)
            final = result
        assert final.completed_updates == 2
        evaluate()
        return observed

    native_samples = {}
    for split in (False, True):
        training.load_checkpoint(checkpoint.id).result(120)
        native_samples[split] = branch(training, split, True)
        training.load_checkpoint(checkpoint.id).result(120)
        replay = branch(training, split, False)
        for before, after in zip(native_samples[split], replay, strict=True):
            np.testing.assert_array_equal(before, after)
    closed(client, training)
    with TemporaryDirectory(prefix="shadowspill-trace-closed-") as directory:
        retained_trace.download(Path(directory) / "trace.zip").result(30)
    receipts.append(
        {
            "run_id": training.run_id,
            "mode": "fresh",
            "step_and_split_oracle": True,
            "evaluation_isolation": True,
            "native_replay_exact": True,
            "runtime_traces": [
                "step",
                "forward_backward",
                "optim_step",
                "forward",
                "masked",
            ],
            "trace_failure_preserves_update": True,
            "trace_download_after_close": True,
        }
    )

    for converted in (False, True):
        if converted:
            run = client.create_training_client(
                model=replace(config.model, initialize=None),
                objective=config.objective,
                optimizer=config.optimizer,
                batch=config.batches,
                memory=config.memory,
                parallelism=config.parallelism,
                execution=replace(config.execution, seed=999),
                operations={
                    "step": ["metrics"],
                    "forward_backward": ["metrics"],
                    "forward": ["predictions"],
                },
                prepare=Preparation(
                    candidates=BuildCandidates(microbatch_sizes=[4]),
                    planning=PlanningOptions(microbatch_counts=[1]),
                    profiling=ProfilingOptions(
                        conditioning_seconds=0.001, measurement_seconds=0.002
                    ),
                ),
                checkpoint=checkpoint,
                name="converted-stochastic",
            ).result(180)
        else:
            run = client.create_training_client(
                plans=plans, checkpoint=checkpoint, name="resumed-stochastic"
            ).result(180)
        info = client._transport.request(
            "GET", "/v1/runs/rng-session/" + run.run_id
        ).value
        assert info.parent_checkpoint == checkpoint.reference
        startup = CheckpointHandle(info.initial_checkpoint, client)
        saved = startup.manifest().result(30)
        if converted:
            mapping = saved.info.conversion.randomness
            assert mapping["root_seed"] == config.execution.seed
            assert {row["action"] for row in mapping["target_streams"]} == {"preserve"}
            assert run.get_config().execution.seed == 999
        for split in (False, True):
            run.load_checkpoint(startup.id).result(120)
            samples = branch(run, split, True)
            if not converted:
                for before, after in zip(native_samples[split], samples, strict=True):
                    np.testing.assert_array_equal(before, after)
            run.load_checkpoint(startup.id).result(120)
            replay = branch(run, split, False)
            for before, after in zip(samples, replay, strict=True):
                np.testing.assert_array_equal(before, after)
        closed(client, run)
        receipts.append(
            {
                "run_id": run.run_id,
                "mode": "converted" if converted else "native",
                "initial_checkpoint": startup.reference.to_dict(),
                "step_and_split_oracle": True,
                "evaluation_isolation": True,
                "replay_exact": True,
                "saved_seed_preserved": True,
                "runtime_traces": [
                    "step",
                    "forward_backward",
                    "optim_step",
                    "forward",
                    "masked",
                ],
            }
        )
    return receipts


def main():
    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, *args):
            if fullname.split(".")[0] in {
                "torch",
                "triton",
                "pynvml",
                "shadowspill",
                "shadowspill_workloads",
            }:
                raise AssertionError(
                    "engine/framework import in demo client: " + fullname
                )

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
        **connection, token=options["token"], session_id="rng-session", timeout=30
    ) as client:
        receipts = qualify(
            client,
            ResolvedTraining.from_dict(options["config"]),
            options["initial"],
            memory,
        )
    assert memory.used == client._memory.used == client._work.pending.used == (0, 0)
    assert not any(name in sys.modules for name in ("torch", "triton", "shadowspill"))
    print(
        json.dumps(
            {
                "client_host": socket.gethostname(),
                "numpy": np.__version__,
                "python": sys.version,
                "runs": receipts,
                "all_client_claims_released": True,
            }
        )
    )


if __name__ == "__main__":
    main()
