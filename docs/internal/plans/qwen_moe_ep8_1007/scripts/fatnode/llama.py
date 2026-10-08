"""Validate symmetric planning with the original 1.18B Llama/FineWeb DP inputs.

Runs through the public Trainer API. Inputs and initialization are read-only;
new plans, metrics, startup timelines, and checks stay in the requested outdir.
"""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import math
import os
import time
from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from mlops.dispatch import set_deterministic_kernels
from mlops.optim import AdamW

from shadowspill.planner import SearchOptions
from shadowspill.pytorch import ProfilingOptions
from shadowspill.training import Distributed, Trainer
from shadowspill.training.backends import ShadowSpill
from shadowspill.training.schedules import WarmupCosine
from workloads.mlops import Llama3, Llama3Config
from workloads.recipes.text.models import build_on_meta


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def objective(model, data):
    tokens, targets, lengths = data
    return model.loss(tokens, targets, seq_lens=lengths, reduction="sum")


def microbatches(data, *, sequences, denominator):
    tokens, targets = data
    for start in range(0, len(tokens), sequences):
        size = min(sequences, len(tokens) - start)
        yield (
            (
                tokens[start : start + size].reshape(1, -1).clone(),
                targets[start : start + size].reshape(1, -1).clone(),
                torch.full((size,), tokens.shape[1], dtype=torch.int32),
            ),
            1.0 / denominator,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-root", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--microbatch-sequences", default="1,2")
    parser.add_argument("--execution-gib", type=float, default=10)
    parser.add_argument("--spill-gib", type=float, default=40)
    parser.add_argument(
        "--symmetric-planning", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    out = args.outdir / f"rank-{rank:05d}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "progress.jsonl").exists():
        raise FileExistsError(f"Fresh validation needs a new output directory: {out}")
    started = time.monotonic()

    def log(event, **details):
        row = dict(
            time=datetime.now(UTC).isoformat(),
            rank=rank,
            event=event,
            elapsed_s=time.monotonic() - started,
            **details,
        )
        print(json.dumps(row), flush=True)
        with (out / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")

    faulthandler.enable()
    faulthandler.dump_traceback_later(900, repeat=True)
    torch.set_num_threads(2)
    set_deterministic_kernels(True)
    manifest = json.loads((args.inputs_root / "inputs.json").read_text())
    config = manifest["config"]
    torch.manual_seed(config["seed"])
    samples = np.load(args.inputs_root / "sequences.npy", mmap_mode="r")
    assert args.steps <= len(samples)
    assert samples.shape[1] % world == 0
    local_sequences = samples.shape[1] // world
    candidates = [int(value) for value in args.microbatch_sequences.split(",")]
    assert all(local_sequences % value == 0 for value in candidates)

    def batch(step):
        first = rank * local_sequences
        values = torch.from_numpy(
            np.array(samples[step, first : first + local_sequences])
        ).long()
        return values[:, :-1].contiguous(), values[:, 1:].contiguous()

    # Assign the shared file mapping directly; only rotary tables need fresh values.
    model = build_on_meta(
        Llama3, dtype=config["compute_dtype"], config=Llama3Config(**manifest["model"])
    ).to_empty(device="cpu")
    model.load_state_dict(
        torch.load(
            args.inputs_root / "initial_weights.pt",
            mmap=True,
            weights_only=True,
            map_location="cpu",
        ),
        assign=True,
    )
    model.rotary.reset_parameters()
    parameters = sum(value.numel() for value in model.parameters())
    assert parameters == manifest["parameters"]
    log(
        "begin",
        world=world,
        parameters=parameters,
        global_tokens=32768,
        local_tokens=32768 // world,
        sequence_length=2048,
        microbatch_tokens=[value * 2048 for value in candidates],
        compute_dtype="float16",
        master_dtype="float32",
        gradient_dtype="float32",
        optimizer_state_dtype="float32",
        symmetric_planning=args.symmetric_planning,
    )
    try:
        with ExitStack() as stack:
            dist.init_process_group("gloo", timeout=timedelta(seconds=1800))
            stack.callback(dist.destroy_process_group)
            if rank == 0:
                for name in ("sequences.npy", "initial_weights.pt"):
                    assert digest(args.inputs_root / name) == manifest["files"][name]
                log("original_inputs_verified", hashes=manifest["files"])
            dist.barrier()
            backend = stack.enter_context(
                ShadowSpill(
                    device=f"cuda:{local_rank}",
                    execution_gib=args.execution_gib,
                    spill_gib=args.spill_gib,
                    control_group=dist.group.WORLD,
                    artifact_store=out / "artifacts",
                    preparation_timeout=1800,
                    search_options=SearchOptions(workers=2),
                    profiling_options=ProfilingOptions(
                        conditioning_seconds=0,
                        measurement_seconds=0,
                        minimum_samples=3,
                    ),
                )
            )
            group = dist.new_group(
                backend="nccl", device_id=backend.device, timeout=timedelta(seconds=600)
            )
            stack.callback(dist.destroy_process_group, group)
            log(
                "runtime_ready",
                device=str(backend.device),
                uuid=str(torch.cuda.get_device_properties(backend.device).uuid),
            )
            trainer = stack.enter_context(
                Trainer(
                    model,
                    objective=objective,
                    optimizer=AdamW,
                    optimizer_args=dict(
                        betas=tuple(config["betas"]),
                        weight_decay=config["weight_decay"],
                        opt_state_dtype=torch.float32,
                        gradient_dtype=torch.float32,
                        parameter_rounding="nearest",
                        opt_state_rounding="nearest",
                    ),
                    schedules={
                        "lr": WarmupCosine(
                            config["learning_rate"],
                            config["min_learning_rate"],
                            config["warmup_steps"],
                            config["steps"],
                        )
                    },
                    master_dtype=torch.float32,
                    grad_dtype=torch.float32,
                    backend=backend,
                    shard_optimizer=True,
                    distributed=Distributed(
                        group,
                        symmetric_planning=args.symmetric_planning,
                        timeout=1800,
                    ),
                    microbatches={
                        f"tokens-{value * 2048}": partial(
                            microbatches,
                            sequences=value,
                            denominator=config["tokens_per_step"],
                        )
                        for value in candidates
                    },
                )
            )
            log("prepare_start")
            trainer.prepare(batch(0))
            trainer.planning.save(out / "search.json")
            write_json(out / "plan.json", trainer.plan.execution_plan.to_dict())
            decisions = [
                json.loads(path.read_text())
                for path in (out / "artifacts").rglob("distributed/*/selection.json")
            ]
            shared = [row for row in decisions if "ordering" in row]
            write_json(out / "symmetric_decisions.json", decisions)
            if args.symmetric_planning:
                assert shared and all(
                    row["planning"]["mode"] == "symmetric" for row in decisions
                ), decisions
            owners = [row.get("searched_by_rank") for row in shared]
            log(
                "prepared",
                selected=trainer.selected_candidate,
                owners=owners,
                ordering_points=len(shared),
                predicted_seconds=trainer.plan.search_result.simulation.makespan_ns
                / 1e9,
            )
            diagnostic = trainer.diagnose(batch(0), directory=out / "startup")
            log("startup_diagnostics_complete", **diagnostic)
            records = []
            historical = [
                json.loads(line)
                for line in (args.inputs_root / "dp1/metrics.jsonl")
                .read_text()
                .splitlines()
            ]
            for step in range(args.steps):
                result = trainer.step(batch(step))
                assert math.isfinite(result.loss)
                gathered = [None] * world
                dist.all_gather_object(
                    gathered, dict(loss=result.loss, seconds=result.seconds)
                )
                total_loss = sum(row["loss"] for row in gathered)
                elapsed = max(row["seconds"] for row in gathered)
                record = dict(
                    step=step + 1,
                    local_loss=result.loss,
                    global_loss=total_loss,
                    slowest_rank_seconds=elapsed,
                    global_tokens_per_second=config["tokens_per_step"] / elapsed,
                    historical_dp1_loss=historical[step]["train/loss"],
                    historical_dp1_loss_difference=total_loss
                    - historical[step]["train/loss"],
                )
                records.append(record)
                log("step_complete", **record)
            state = trainer._execution.call.state_dict(weights="compute")["model"]
            hashes = {}
            original = torch.load(
                args.inputs_root / "initial_weights.pt",
                mmap=True,
                weights_only=True,
                map_location="cpu",
            )
            changed = []
            for name, _ in model.named_parameters():
                value = state[name]
                assert torch.isfinite(value).all(), name
                hashes[name] = hashlib.sha256(
                    value.contiguous().numpy().tobytes()
                ).hexdigest()
                if not torch.equal(value, original[name]):
                    changed.append(name)
            replicas = [None] * world
            dist.all_gather_object(replicas, hashes)
            assert all(value == hashes for value in replicas), "Final replicas differ"
            assert changed, "No model parameters changed"
            write_json(out / "parameter_hashes.json", hashes)
            result = dict(
                passed=True,
                world=world,
                rank=rank,
                parameters=parameters,
                symmetric_planning=args.symmetric_planning,
                owners=owners,
                selected_candidate=trainer.selected_candidate,
                steps=records,
                replicas_bitwise_equal=True,
                changed_parameter_tensors=len(changed),
                historical_reference="Original DP1 run; accumulation order may differ",
            )
            write_json(out / "result.json", result)
            log(
                "passed",
                replicas_bitwise_equal=True,
                changed_parameter_tensors=len(changed),
            )
    except BaseException as error:
        write_json(
            out / "failure.json",
            dict(error=repr(error), elapsed_s=time.monotonic() - started),
        )
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
