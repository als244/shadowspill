"""Real-data public training qualification for Qwen pretraining and GLM LoRA.

Run inside an existing eight-GPU allocation. ``--preflight`` is CPU-only and
checks the exact settings, layouts and retained corpus before requesting GPUs.
This is a learning/reporting check, separate from the compiled numerical gates.
It deliberately saves no checkpoints or weight exports.
"""

import argparse
import json
import math
import os
import secrets
import socket
import sys
import traceback
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic

import numpy as np

from examples.training.text_data import TokenBatches
from shadowspill_client import (
    BatchConfig,
    LoRA,
    Memory,
    Model,
    Objective,
    Optimizer,
    Preparation,
    ServiceClient,
    Weights,
)
from shadowspill_client.display import CommandTable
from shadowspill_client.parallelism import data_parallel, expert_parallel
from shadowspill_client.protocol.setup import TrainingSetup
from shadowspill_client.text import build_candidates, text_batch_statistics


def settings(kind="qwen", parallelism="dp", checkpoint=None):
    """Same model, optimizer, batch, budgets and build choices as the README."""
    result = {
        "model": Model(
            "qwen3moe", config={"preset": "30b-a3b"}, initialize=Weights.random(seed=0)
        ),
        "objective": Objective.sum(
            {"tokens": Objective("cross_entropy"), "balance": Objective("moe_balance")},
            weights={"balance": 0.001},
        ),
        "optimizer": Optimizer(
            "adamw",
            precision={
                "gradient_dtype": "float32",
                "opt_state_dtype": "float32",
                "master_dtype": "float32",
                "parameter_rounding": "stochastic",
                "opt_state_rounding": "stochastic",
            },
        ),
        "parallelism": data_parallel(8),
        "batch": BatchConfig(
            limits={
                "token_capacity": 2097152,
                "max_sequence_length": 4096,
                "max_sequences": 4096,
            }
        ),
        "memory": Memory(device_gib=20, spill_gib=96),
        "prepare": Preparation(
            candidates=build_candidates(
                microbatch_tokens=[65536],
                projection_chunks=[0],
            )
        ),
    }
    if parallelism == "ep":
        result["model"] = Model(
            "qwen3moe",
            config={"preset": "30b-a3b"},
            implementation="mlops_ep",
            initialize=Weights.random(seed=0),
        )
        result["parallelism"] = expert_parallel(
            8,
            experts=[
                "blocks.*.moe.experts.gate_up_weight",
                "blocks.*.moe.experts.down_weight",
            ],
        )
    if kind == "glm":
        if checkpoint is None:
            raise ValueError("GLM qualification requires a real pretrained checkpoint")
        result.update(
            optimizer=Optimizer(
                "adamw",
                precision={
                    "gradient_dtype": "bfloat16",
                    "opt_state_dtype": "bfloat16",
                    "master_dtype": "bfloat16",
                    "parameter_rounding": "stochastic",
                    "opt_state_rounding": "stochastic",
                },
            ),
            model=Model.from_pretrained(
                str(checkpoint),
                lora=LoRA(32, alpha=32),
                implementation="mlops_ep" if parallelism == "ep" else "mlops",
            ),
            objective=Objective("cross_entropy"),
            memory=Memory(device_gib=40, spill_gib=96),
            batch=BatchConfig(
                limits={
                    "token_capacity": 262144,
                    "max_sequence_length": 32768,
                    "max_sequences": 64,
                }
            ),
            prepare=Preparation(candidates=build_candidates(microbatch_tokens=[32768])),
        )
        if parallelism == "ep":
            result["parallelism"] = expert_parallel(
                8, experts=["layers.*.mlp.experts.*"]
            )
    return result


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def preflight(data, output, requested, *, cache_root=None):
    from shadowspill.checkpoint.file_digest import checksum_cache
    from shadowspill.training.definitions import load_registry
    from shadowspill.training.preparation.capture.templates import (
        capture_inventory,
    )
    from shadowspill.training.preparation.configuration.layouts import prepare_layouts
    from shadowspill.training.preparation.setup import resolve_training

    batch = data[0]
    limits = requested["batch"].limits
    assert sum(len(d.inputs["tokens"]) for d in batch) == limits["token_capacity"]
    assert all(np.max(d.inputs["tokens"]) < data.vocab_size for d in batch)
    with checksum_cache(None if cache_root is None else cache_root / "source-digests"):
        setup = resolve_training(
            load_registry(["shadowspill", "shadowspill-workloads"]),
            TrainingSetup(**requested),
        )
    layouts = prepare_layouts(setup)
    inventory = capture_inventory(layouts)
    write_json(
        output / "preflight.json",
        {
            "status": "PASS",
            "host": socket.gethostname(),
            "data": data.identity,
            "batch": {
                "sequences": len(batch),
                "tokens": limits["token_capacity"],
                "length": limits["max_sequence_length"],
            },
            "build_cases": len(setup.config.preparation.cases),
            "microbatch_counts": list(
                setup.config.preparation.planning.microbatch_counts
            ),
            "templates": len(inventory.templates),
            "staging_bytes": {
                str(k): v.staging_bytes for k, v in layouts.staging.items()
            },
        },
    )
    write_json(output / "requested.json", TrainingSetup(**requested).to_dict())
    print(setup.preview().summary(), flush=True)
    return batch


def report_document(path):
    return json.loads(
        path.read_text()
        .split('<script id="data" type="application/json">', 1)[1]
        .split("</script>", 1)[0]
    )


def train(service, batch, data, output, steps, requested, lr):
    receipt = {
        "host": socket.gethostname(),
        "data": data.identity,
        "steps": [],
        "checkpoint_saved": False,
        "started_utc": datetime.now(UTC).isoformat(),
    }
    start = monotonic()
    preview = service.describe_training(**requested).result(300)
    write_json(output / "preview.json", preview.to_dict())
    measured = service.prepare_training(config=preview.config, progress=True).result(
        14400
    )
    measured.export().result().download(output / "measurements.zip").result(600)
    write_json(
        output / "measurement-reuse.json",
        {
            "measurements": {
                "session_id": measured.info.session_id,
                "run_id": measured.info.run_id,
            },
        },
    )
    training = service.create_training_client(
        **{**requested, "prepare": replace(requested["prepare"], reuse=measured.reuse)},
        progress=True,
    ).result(14400)
    receipt.update(run_id=training.run_id, preparation_seconds=monotonic() - start)
    write_json(output / "receipt.json", receipt)
    try:
        training.plans.download(output / "plans.ssp").result(600)
        write_json(output / "plans.json", training.plans.reference.to_dict())
        training.plans.report().result(300).download(output / "plan.html").result(300)
        assert report_document(output / "plan.html")["views"]
        receipt["plan_report_before_training"] = True
        validation = training.validate_batch(batch).result(300)
        assert validation.packing_checked and validation.estimated_seconds is not None
        write_json(
            output / "validation.json",
            {
                "operation": validation.operation.to_dict(),
                "capacities": dict(validation.capacities),
                "packing_checked": validation.packing_checked,
                "candidate": validation.candidate,
                "scheduled_capacities": dict(validation.scheduled_capacities),
                "estimated_seconds": validation.estimated_seconds,
            },
        )
        table = CommandTable()
        stats = text_batch_statistics(batch)

        def finish(index, future):
            result = future.result(3600)
            assert (
                result.outcome == "completed" and result.completed_updates == index + 1
            )
            loss = float(result.loss)
            assert math.isfinite(loss) and 0 < loss < 2 * math.log(data.vocab_size), (
                loss
            )
            assert result.execution_seconds is not None and result.execution_seconds > 0
            assert all(math.isfinite(float(v)) for v in result.metrics.values())
            if "terms" in requested["objective"].config:
                assert (
                    "tokens.loss_sum" in result.metrics
                    and "balance.loss_sum" in result.metrics
                )
            receipt["steps"].append(result.record.to_dict())
            write_json(output / "receipt.json", receipt)
            if index == steps - 1:
                assert result.traces
                result.traces[0].download_report(output / "runtime.html").result(300)
                assert (
                    report_document(output / "runtime.html")["kind"] == "runtime_trace"
                )

        pending = None
        for index in range(steps):
            following = training.step(
                batch, optim={"lr": lr}, runtime_trace=index == steps - 1
            )
            table.track(
                "step",
                following,
                batch=stats,
                planned_seconds=validation.estimated_seconds,
            )
            if pending is not None:
                finish(*pending)
            pending = index, following
        finish(*pending)
        losses = [float(row["loss"]) for row in receipt["steps"]]
        receipt["fixed_batch_learning"] = losses[-1] < losses[0]
        assert receipt["fixed_batch_learning"], losses
        receipt["status"] = "PASS"
        receipt["predicted_seconds"] = validation.estimated_seconds
        receipt["client_framework_free"] = not any(
            name in sys.modules for name in ("torch", "triton", "mlops")
        )
        assert receipt["client_framework_free"]
        write_json(output / "receipt.json", receipt)
        print(f"Fixed-batch loss: {losses[0]:.5f} → {losses[-1]:.5f}", flush=True)
    finally:
        training.close().result(300)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--service-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--model", choices=("qwen", "glm"), default="qwen")
    parser.add_argument("--parallelism", choices=("dp", "ep"), default="dp")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.steps < 2:
        parser.error("at least two steps are needed to check learning")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.model == "glm" and args.checkpoint is None:
        parser.error("GLM requires --checkpoint")
    requested = settings(args.model, args.parallelism, args.checkpoint)
    limits = requested["batch"].limits
    data = TokenBatches(
        args.data,
        sequence_length=limits["max_sequence_length"],
        sequences_per_step=limits["token_capacity"] // limits["max_sequence_length"],
    )
    cache_root = args.cache_root or (
        (args.service_root or args.output).parent / "cache"
    )
    batch = preflight(data, args.output, requested, cache_root=cache_root)
    if args.preflight:
        return
    if args.service_root is None:
        parser.error("--service-root must name a local SSD directory")
    from shadowspill.server import Server

    token = secrets.token_urlsafe(32)
    os.environ["SHADOWSPILL_TOKEN"] = token
    try:
        with Server(
            service_root=args.service_root,
            cache_root=cache_root,
            model_root=None
            if args.checkpoint is None
            else args.checkpoint.resolve().parent,
        ) as server:
            server.start(port=0)
            with ServiceClient(
                f"http://127.0.0.1:{server.address[1]}", token=token
            ) as client:
                train(
                    client,
                    batch,
                    data,
                    args.output,
                    args.steps,
                    requested,
                    1e-4 if args.model == "glm" else 3e-4,
                )
    except BaseException:
        write_json(
            args.output / "failure.json",
            {
                "failed_utc": datetime.now(UTC).isoformat(),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()
