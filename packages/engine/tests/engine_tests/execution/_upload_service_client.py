"""NumPy-only checkpoint transfer and continued whole/split training oracle."""

import importlib.abc
import json
import runpy
import socket
import sys
import tarfile
from pathlib import Path
from tempfile import mkdtemp

import numpy as np

from shadowspill_client import SSH, ServiceClient
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.requests import ArtifactRef
from shadowspill_client.protocol.training import ResolvedTraining


def main():
    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, *args):
            if fullname.split(".")[0] in {
                "torch",
                "triton",
                "shadowspill",
                "shadowspill_workloads",
            }:
                raise AssertionError("framework import in transfer client: " + fullname)

    sys.meta_path.insert(0, Block())
    helpers = runpy.run_path(str(Path(__file__).with_name("_rng_service_client.py")))
    Oracle, records, outputs, closed = (
        helpers[name] for name in ("Oracle", "records", "outputs", "closed")
    )
    raw = sys.stdin.buffer.read((1 << 20) + 1)
    assert len(raw) <= 1 << 20
    options = json.loads(raw)
    trainable = options["config"]["model"]["trainable"]
    connection = (
        {"ssh": SSH(options["ssh_host"], options["port"])}
        if options.get("ssh_host")
        else {"base_url": f"http://127.0.0.1:{options['port']}"}
    )
    memory = BufferBudget(64 << 20, 8)
    with ServiceClient(
        **connection, token=options["token"], session_id="rng-session"
    ) as client:
        if options["phase"] == "source":
            training = client.create_training_client(
                config=ResolvedTraining.from_dict(options["config"])
            ).result(180)
            oracle = Oracle(options["initial"], trainable=trainable)
            first = training.step(records(), optim={"lr": 0.03}, grad_scale=0.5).result(
                120
            )
            oracle.check(outputs(first, memory), first, backward=True)
            oracle.update(0.03, 0.5)
            assert first.completed_updates == 1
            checkpoint = training.save_checkpoint(
                name="portable", metadata={"cursor": 3}
            ).result(120)
            manifest = checkpoint.manifest().result(30)
            directory = Path(mkdtemp(prefix="shadowspill-training-upload-"))
            checkpoint.download(directory / "checkpoint").result(60)
            assert (manifest.info.frozen_base is not None) == (trainable is not None)
            if trainable is not None:
                assert manifest.info.frozen_base.model.trainable == tuple(trainable)
                for path in (directory / "checkpoint" / "ranks").iterdir():
                    with tarfile.open(path) as archive:
                        rank = json.loads(archive.extractfile("manifest.json").read())
                    assert rank["omitted_parameters"] == [
                        "encoder.bias",
                        "encoder.weight",
                    ]
                    assert not {"encoder.bias", "encoder.weight"}.intersection(
                        row["name"] for row in rank["tensors"]
                    )
            dependencies = (manifest.info.configuration, manifest.info.plans)
            for reference in dependencies:
                client.artifact(reference).download(directory / reference.id).result(60)
            state = {
                "checkpoint": checkpoint.reference.to_dict(),
                "dependencies": [ref.to_dict() for ref in dependencies],
                "boundary_weights": {
                    name: value.tolist() for name, value in oracle.weights.items()
                },
            }
        else:
            directory = Path(options["backup"])
            state = json.loads((directory / "state.json").read_text())
            checkpoint = client.artifacts.upload(
                directory / "checkpoint",
                reference=ArtifactRef.from_dict(state["checkpoint"]),
            ).result(60)
            dependencies = tuple(
                ArtifactRef.from_dict(value) for value in state["dependencies"]
            )
            uploaded = [
                client.artifacts.upload(directory / ref.id, reference=ref).result(60)
                for ref in dependencies
            ]
            # No explicit config and no copied/generated Plan summaries.
            training = client.create_training_client(
                plans=uploaded[-1], checkpoint=checkpoint
            ).result(180)
            info = client._transport.request(
                "GET", f"/v1/runs/{training.session_id}/{training.run_id}"
            ).value
            assert info.initial_checkpoint == checkpoint.reference
            assert info.initial_completed_updates == 1
            oracle = Oracle(state["boundary_weights"], trainable=trainable)

        observations = []

        def finish(future, *, temperature=1.0, backward=False):
            result = future.result(120)
            assert result.state_valid and not result.artifact_issues
            values = outputs(result, memory) if result.outputs else {}
            if values:
                oracle.check(values, result, temperature=temperature, backward=backward)
            observations.append(
                {
                    "operation": result.operation,
                    "outcome": result.outcome,
                    "updates": result.completed_updates,
                    "loss": result.loss,
                    "outputs": {key: value.tolist() for key, value in values.items()},
                }
            )
            return result

        whole = finish(
            training.step(
                records(),
                optim={"lr": 0.04},
                grad_scale=1.25,
                loss_params={"temperature": 1.5},
            ),
            temperature=1.5,
            backward=True,
        )
        oracle.update(0.04, 1.25)
        assert whole.completed_updates == 2
        finish(
            training.forward_backward(records(), loss_params={"temperature": 0.5}),
            temperature=0.5,
            backward=True,
        )
        finish(training.forward(records(targets=False), outputs="predictions"))
        masked = finish(training.forward_backward(records(masked=True)))
        assert masked.outcome == "skipped" and masked.completed_updates == 2
        finish(training.forward_backward(records()), backward=True)
        split = finish(training.optim_step(optim={"lr": 0.02}, grad_scale=-0.5))
        oracle.update(0.02, -0.5)
        assert split.completed_updates == 3
        finish(training.forward(records(targets=False), outputs="predictions"))
        if options["phase"] == "source":
            state["observations"] = observations
            state["final_weights"] = {
                name: value.tolist() for name, value in oracle.weights.items()
            }
            (directory / "state.json").write_text(json.dumps(state) + "\n")
        else:
            for expected, actual in zip(
                state["observations"], observations, strict=True
            ):
                for key in ("operation", "outcome", "updates"):
                    assert actual[key] == expected[key]
                for name, value in actual["outputs"].items():
                    if name == "noise":
                        np.testing.assert_array_equal(value, expected["outputs"][name])
                    else:
                        np.testing.assert_allclose(
                            value, expected["outputs"][name], rtol=2e-4, atol=2e-5
                        )
                if actual["loss"] is not None:
                    np.testing.assert_allclose(
                        actual["loss"], expected["loss"], rtol=2e-4, atol=2e-5
                    )
            for name, value in oracle.weights.items():
                np.testing.assert_array_equal(
                    value, np.asarray(state["final_weights"][name], dtype=np.float32)
                )
        if trainable is not None:
            restored = training.load_checkpoint(checkpoint.id).result(120)
            assert restored.completed_updates == 1
            oracle = Oracle(state["boundary_weights"], trainable=trainable)
            again = training.step(
                records(),
                optim={"lr": 0.04},
                grad_scale=1.25,
                loss_params={"temperature": 1.5},
            ).result(120)
            values = outputs(again, memory)
            oracle.check(values, again, temperature=1.5, backward=True)
            np.testing.assert_array_equal(
                values["noise"], observations[0]["outputs"]["noise"]
            )
            assert again.completed_updates == 2
            following = training.save_checkpoint().result(120)
            before = checkpoint.manifest().result(30)
            after = following.manifest().result(30)
            assert before.info.frozen_base == after.info.frozen_base
        closed(client, training)
    assert memory.used == client._memory.used == client._work.pending.used == (0, 0)
    assert not any(name in sys.modules for name in ("torch", "triton", "shadowspill"))
    print(
        json.dumps(
            {
                "phase": options["phase"],
                "backup": str(directory),
                "client_host": socket.gethostname(),
                "all_client_claims_released": True,
                "checkpoint": state["checkpoint"],
                "run_id": training.run_id,
                "independent_step_and_split_oracle": True,
                "exact_training_and_evaluation_rng": options["phase"] == "target",
                "frozen_base_reconstructed": trainable is not None,
                "observations": observations,
            }
        )
    )


if __name__ == "__main__":
    main()
