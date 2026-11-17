"""Client-only text training checks; independent reference values arrive as data."""

import importlib.abc
import json
import socket
import sys
import tarfile
from dataclasses import replace
from math import prod
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from time import monotonic

import numpy as np

from shadowspill_client import (
    SSH,
    BatchConfig,
    BuildCandidates,
    BuildOptions,
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


def exported(training, trained, initial, *, separate_masters=True):
    handle = training.export_weights().result(120)
    with TemporaryDirectory(prefix="shadowspill-llama-weights-") as directory:
        root = handle.download(Path(directory) / "weights").result(60)
        return read_weights(root, trained, initial, separate_masters=separate_masters)


def read_weights(root, trained, initial, *, separate_masters=True):
    """Independent NumPy reconstruction of small qualification exports.

    Saved logical coordinates describe expert/model shards; storage regions
    describe flat optimizer shards. Check coverage and equal replicas rather
    than assuming every rank contains a complete contiguous parameter.
    """
    manifests = []
    for path in sorted((root / "ranks").glob("rank-*.ssp")):
        with tarfile.open(path) as archive:
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
            manifests.append((manifest, arrays))
    assert manifests, "weights export contains no ranks"
    masters = {
        row["tensor"]["identity"]["name"]
        for manifest, _ in manifests
        for row in manifest["model_tensors"]
        if row["tensor"]["identity"]["role"] == "master"
    }
    assert masters == (set(trained) if separate_masters else set()), (masters, trained)
    values, coverage = {}, {}
    for manifest, arrays in manifests:
        for row in manifest["model_tensors"]:
            view, identity = row["view"], row["tensor"]["identity"]
            name, role = identity["name"], identity["role"]
            if name not in initial or role != (
                "master" if name in masters else "parameter"
            ):
                continue
            assert not identity["component_path"], "encoded values need an explicit map"
            shape = tuple(view["shape"])
            assert shape == np.shape(initial[name]), name
            if name not in values:
                values[name] = np.empty(shape, dtype=np.float32)
                coverage[name] = np.zeros(shape, dtype=bool)
            if row["archive"] in manifest["omitted_parameters"]:
                assert manifest["frozen_state_id"] is not None and name not in trained
                values[name][...] = initial[name]
                coverage[name][...] = True
                continue
            if not view["region_elements"]:
                continue
            array = arrays[
                manifest["derived_tensors"].get(row["archive"], row["archive"])
            ]
            assert role != "master" or (array.dtype.kind == "f" and array.itemsize == 4)
            # Enumerate only these small test models; production reconstruction
            # uses the engine's bounded readers and span intersections instead.
            indices = np.indices(view["local_shape"]).reshape(
                len(shape), prod(view["local_shape"])
            )
            offsets = (
                view["storage_offset"]
                + np.asarray(view["stride"], dtype=np.int64) @ indices
            )
            offsets -= view["region_offset"]
            selected = (offsets >= 0) & (offsets < view["region_elements"])
            coordinates = tuple(
                indices[:, selected] + np.asarray(view["origin"])[:, None]
            )
            incoming = array[offsets[selected]]
            previous = coverage[name][coordinates]
            np.testing.assert_array_equal(
                values[name][coordinates][previous], incoming[previous]
            )
            values[name][coordinates] = incoming
            coverage[name][coordinates] = True
    assert values.keys() == initial.keys(), (values.keys(), initial.keys())
    assert all(mask.all() for mask in coverage.values()), "incomplete logical weights"
    return values


def logprobs(training, records, budget, output_name="logprobs"):
    values = [
        Datum(row["inputs"], {"target_tokens": row["loss_inputs"]["target_tokens"]})
        for row in records
    ]
    result = training.forward(values, outputs="logprobs").result(120)
    assert result.loss is None
    handle = result.outputs[output_name]
    with TemporaryDirectory(prefix="shadowspill-llama-logprobs-") as directory:
        path = handle.download(Path(directory) / "output.ssp").result(60)
        with (
            path.open("rb") as stream,
            read_payload(stream, handle.reference, budget) as decoded,
        ):
            fields = decoded.datums[0].inputs
            return [fields[str(index)].array().copy() for index in range(len(records))]


def check(training, options, index, budget, *, expected_initial=None):
    expected = options["oracle"][index]
    initial = options["oracle"][0]["weights"]
    actual = exported(
        training,
        options["trainable"],
        initial,
        separate_masters=options.get("separate_fp32_masters", True),
    )
    for name, value in expected["weights"].items():
        value = np.asarray(value, dtype=np.float32)
        if index == 0 and expected_initial is not None:
            np.testing.assert_array_equal(actual[name], expected_initial[name])
        elif index == 0 or name not in options["trainable"]:
            np.testing.assert_array_equal(actual[name], value)
        else:
            # Compare updates, so large unchanged weights cannot hide missing or
            # incorrectly scaled updates. Independent eager activation derivatives
            # and fused GPU kernels can still differ at BF16 rounding boundaries.
            baseline = np.asarray(initial[name], dtype=np.float32)
            actual_delta, expected_delta = actual[name] - baseline, value - baseline
            absolute = options.get("weight_atol", 1e-5)
            if "weight_norm_rtol" in options:
                # Signed token contributions can nearly cancel individual BF16
                # derivatives. Bound every parameter's complete update error,
                # and separately reject outliers relative to its largest update.
                error = np.linalg.norm(actual_delta - expected_delta)
                bound = options["weight_norm_rtol"] * np.linalg.norm(expected_delta)
                assert error <= bound + 1e-7, (name, error, bound)
                absolute += options["weight_max_rtol"] * np.max(np.abs(expected_delta))
            np.testing.assert_allclose(
                actual_delta,
                expected_delta,
                rtol=options.get("weight_rtol", 0.03),
                atol=absolute,
                err_msg=name,
            )
    check_logprobs(training, options, index, budget)
    return actual


def check_logprobs(training, options, index, budget):
    for actual_row, expected_row in zip(
        logprobs(
            training, options["records"], budget, options.get("output_name", "logprobs")
        ),
        options["oracle"][index]["logprobs"],
        strict=True,
    ):
        np.testing.assert_allclose(
            actual_row,
            expected_row,
            rtol=options.get("logprob_rtol", 0.003),
            atol=options.get("logprob_atol", 0.008),
        )


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


def training_settings(options):
    return dict(
        model=Model.from_dict(options["model"]),
        objective=Objective.from_dict(options["objective"]),
        optimizer=Optimizer(
            "adamw",
            defaults=options["optim"],
            precision={
                "gradient_dtype": options.get("gradient_dtype", "float32"),
                "opt_state_dtype": "float32",
                "master_dtype": "float32",
            },
        ),
        batch=BatchConfig(
            limits=options.get(
                "batch_limits",
                {"token_capacity": 8, "max_sequence_length": 4, "max_sequences": 4},
            )
        ),
        memory=Memory(options.get("device_gib", 2), 0.125),
        parallelism=options.get("parallelism"),
        operations={
            "step": ["metrics"],
            "forward_backward": ["metrics"],
            "forward": ["logprobs"],
        },
        prepare=Preparation(
            build=BuildOptions.from_dict(options.get("build", {})),
            candidates=BuildCandidates(
                microbatch_sizes=options.get("microbatch_sizes", [4, 8])
            ),
            planning=PlanningOptions(
                microbatch_counts=options.get("microbatch_counts", [1, 2])
            ),
            profiling=ProfilingOptions(
                conditioning_seconds=0.001, measurement_seconds=0.002
            ),
        ),
    )


def check_loss_terms(result, options):
    """Public term metrics survive transport and retain their own reductions."""
    names = options.get("loss_terms", ())
    if not names:
        return
    total = 0.0
    for name in names:
        loss = result.metrics[f"{name}.loss_sum"]
        count = result.metrics[f"{name}.valid_tokens"]
        assert np.isfinite(loss) and count > 0
        total += result.loss_params[f"{name}.weight"] * loss
    np.testing.assert_allclose(result.loss, total, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(result.metrics["loss_sum"], total, rtol=2e-6, atol=2e-6)


def check_rejected_training(training, records, controls):
    """Globally masked inputs fail local validation and submission alike.

    An empty input owner inside a participating distributed batch is different:
    the split-update checks above exercise that supported case.
    """
    for operation in ("step", "forward_backward"):
        for validate_only in (True, False):
            try:
                if validate_only:
                    training.validate_batch(records, operation=operation).result(120)
                else:
                    getattr(training, operation)(records, loss_params=controls).result(
                        120
                    )
            except ValueError as error:
                assert "no loss participation" in str(error), error
            else:
                raise AssertionError(f"all-masked {operation} was accepted")


def qualify(client, options, budget):
    settings = training_settings(options)
    preview = client.describe_training(**settings).result(30)
    training = client.create_training_client(config=preview.config).result(
        options.get("preparation_timeout", 480)
    )
    plans = training.plans
    initial = training.save_checkpoint(name="initial").result(120)
    records = [Datum(**row) for row in options["records"]]
    observations, whole = [], None
    base_options = options
    for split in (False, True):
        options = base_options
        if "oracles" in base_options:
            options = {
                **base_options,
                "oracle": base_options["oracles"]["split" if split else "whole"],
            }
        if split:
            training.load_checkpoint(initial.id).result(120)
        check(training, options, 0, budget)
        if split:
            for part, subset in enumerate((records[:1], records[1:])):
                result = training.forward_backward(
                    subset, loss_params=options["controls"]
                ).result(120)
                check_loss_terms(result, options)
                np.testing.assert_allclose(
                    result.loss,
                    options["loss_parts"][part],
                    rtol=options.get("loss_rtol", 0.003),
                    atol=options.get("loss_atol", 0.008),
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
            check_loss_terms(result, options)
            np.testing.assert_allclose(
                result.loss,
                options.get("whole_loss", sum(options["loss_parts"])),
                rtol=options.get("loss_rtol", 0.003),
                atol=options.get("loss_atol", 0.008),
            )
        assert result.outcome == "completed" and result.completed_updates == 1
        observations.append(result.record.to_dict())
        actual = check(training, options, 1, budget)
        if whole is None:
            whole = actual
        elif options.get("whole_split_equivalence", True):
            for name in options["trainable"]:
                np.testing.assert_allclose(
                    actual[name], whole[name], rtol=1e-4, atol=3e-6, err_msg=name
                )
    checkpoint = training.save_checkpoint(name="after-one").result(120)
    weights_artifact = None
    source_weights = actual
    if base_options.get("weights_initialization") is not None:
        weights_artifact = training.export_weights().result(120)
    masked = [
        Datum(
            row.inputs,
            {**row.loss_inputs, options["mask"]: [False] * len(row.inputs["tokens"])},
        )
        for row in records
    ]
    check_rejected_training(training, masked, options["controls"])
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
        if options.get("exact_resume", False):
            np.testing.assert_array_equal(actual[name], final[name], err_msg=name)
        else:
            np.testing.assert_allclose(
                actual[name], final[name], rtol=1e-4, atol=3e-6, err_msg=name
            )
    close(client, resumed)
    runs = [training.run_id, resumed.run_id]
    fresh_options = base_options.get("fresh_saved_plans") or base_options.get(
        "weights_initialization"
    )
    if fresh_options is not None:
        model = Model.from_dict(fresh_options["model"])
        expected_initial = None
        if weights_artifact is not None:
            model = replace(model, initialize=Weights.artifact(weights_artifact.id))
            expected_initial = source_weights
        fresh = client.create_training_client(
            plans=plans,
            **{**settings, "model": model},
        ).result(240)
        assert fresh.plans.reference != plans.reference
        assert fresh.get_config().model.initialize == model.initialize
        fresh_initial = fresh.save_checkpoint(name="fresh-initial").result(120)
        assert fresh_initial.manifest().result().info.completed_updates == 0
        for split in (True, False) if weights_artifact is not None else (True,):
            if not split:
                fresh.load_checkpoint(fresh_initial.id).result(120)
            check(fresh, fresh_options, 0, budget, expected_initial=expected_initial)
            if split:
                for part, subset in enumerate((records[:1], records[1:])):
                    result = fresh.forward_backward(
                        subset, loss_params=fresh_options["controls"]
                    ).result(120)
                    np.testing.assert_allclose(
                        result.loss,
                        fresh_options["loss_parts"][part],
                        rtol=fresh_options.get("loss_rtol", 0.003),
                        atol=fresh_options.get("loss_atol", 0.008),
                    )
                    check_logprobs(fresh, fresh_options, 0, budget)
                result = fresh.optim_step(grad_scale=0.25).result(120)
            else:
                result = fresh.step(
                    records, loss_params=fresh_options["controls"], grad_scale=0.25
                ).result(120)
                np.testing.assert_allclose(
                    result.loss,
                    sum(fresh_options["loss_parts"]),
                    rtol=fresh_options.get("loss_rtol", 0.003),
                    atol=fresh_options.get("loss_atol", 0.008),
                )
            assert result.completed_updates == 1
            check(fresh, fresh_options, 1, budget)
        saved = fresh.save_checkpoint(name="fresh-after-one").result(120)
        check_rejected_training(fresh, masked, fresh_options["controls"])
        check(fresh, fresh_options, 1, budget)
        fresh.step(records, loss_params=fresh_options["controls"]).result(120)
        expected = check(fresh, fresh_options, 2, budget)
        close(client, fresh)
        continued = client.create_training_client(
            plans=fresh.plans, checkpoint=saved
        ).result(240)
        check(continued, fresh_options, 1, budget)
        continued.step(records, loss_params=fresh_options["controls"]).result(120)
        actual = check(continued, fresh_options, 2, budget)
        for name in fresh_options["trainable"]:
            np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)
        close(client, continued)
        runs.extend((fresh.run_id, continued.run_id))
    return {
        "runs": runs,
        "plans": plans.reference.to_dict(),
        "observations": observations,
        "whole_split_equivalence": options.get("whole_split_equivalence", True),
        "independent_geometry_oracles": "oracles" in options,
        "evaluation_isolation": True,
        "zero_participation_rejected": True,
        "adamw_resume": True,
        "separate_fp32_masters": options.get("separate_fp32_masters", True),
        "same_dtype_master_reuse": not options.get("separate_fp32_masters", True),
        "exact_resume": options.get("exact_resume", False),
        "fresh_saved_plans": fresh_options is not None,
        "weights_initialization": weights_artifact is not None,
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
    # EP has minimum aligned widths and includes every logical expert's weights
    # in both trajectories. This private fixture is larger than service inputs.
    # This bounds test-fixture input, independently of service request limits.
    raw = sys.stdin.buffer.read((64 << 20) + 1)
    assert len(raw) <= 64 << 20
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
