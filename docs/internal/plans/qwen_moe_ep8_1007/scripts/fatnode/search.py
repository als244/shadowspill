"""Real DP microbatch/order search, then compare trained weights with a CPU oracle."""

import argparse
import json
import os
import time
from datetime import timedelta
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist

from shadowspill.pytorch import ProfilingOptions
from shadowspill.search.geometries import default_orderings
from shadowspill.training import Distributed, Trainer
from shadowspill.training.backends import ShadowSpill
from tests.shadowspill.pytorch.distributed._training_case import (
    Model,
    objective,
    snapshot,
)


class SearchModel(Model):
    def forward(self, x):
        return self.second(torch.tanh(self.first(x)))


def microbatches(data, rows):
    x, y, denominator = data
    for start in range(0, len(x), rows):
        yield (
            (
                x[start : start + rows].clone(),
                y[start : start + rows].clone(),
                denominator,
            ),
            1.0,
        )


rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--partition", choices=("auto", "whole"), default="auto")
parser.add_argument("--mutable-buffer", action="store_true")
parser.add_argument("--rows-per-rank", type=int, default=4)
parser.add_argument("--outdir", type=Path)
parser.add_argument(
    "--symmetric-planning", action=argparse.BooleanOptionalAction, default=True
)
args = parser.parse_args()
label = "capture-control" if args.mutable_buffer else "sweep-" + args.partition
root = args.outdir or (
    Path(__file__).resolve().parents[2] / ("evidence/fatnode/" + label)
)
model_type = Model if args.mutable_buffer else SearchModel
out = root / f"rank-{rank:05d}"
out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)


def data(peer, step):
    generator = torch.Generator().manual_seed(1234 + peer * 101 + step)
    return (
        torch.randn(args.rows_per_rank, 7, generator=generator),
        torch.randn(args.rows_per_rank, 5, generator=generator),
        float(args.rows_per_rank * world * 5),
    )


dist.init_process_group("gloo", timeout=timedelta(seconds=300))
try:
    with ShadowSpill(
        device="auto",
        execution_gib=2,
        spill_gib=2,
        control_group=dist.group.WORLD,
        preparation_timeout=300,
        artifact_store=out / "artifacts",
        partition=args.partition,
        profiling_options=ProfilingOptions(
            conditioning_seconds=0, measurement_seconds=0, minimum_samples=3
        ),
    ) as backend:
        group = dist.new_group(
            backend="nccl", device_id=backend.device, timeout=timedelta(seconds=300)
        )
        try:
            options = dict(
                lr=0.005, betas=(0.8, 0.95), eps=1e-4, weight_decay=0.0, foreach=False
            )
            reference = model_type()
            optimizer = torch.optim.AdamW(reference.parameters(), **options)
            with Trainer(
                model_type(),
                objective=objective,
                optimizer=torch.optim.AdamW,
                optimizer_args=options,
                backend=backend,
                distributed=Distributed(
                    group, symmetric_planning=args.symmetric_planning, timeout=300
                ),
                grad_dtype=torch.float32,
                shard_optimizer=True,
                microbatches={
                    "rows2": partial(microbatches, rows=2),
                    "rows1": partial(microbatches, rows=1),
                },
            ) as trainer:
                started = time.monotonic()
                trainer.prepare(data(rank, 0))
                trainer.planning.save(out / "search.json")
                search = [
                    json.loads(p.read_text())
                    for p in (out / "artifacts").rglob("distributed/*/selection.json")
                ]
                shared = [row for row in search if "ordering" in row]
                ordering_counts = [
                    len(default_orderings(args.rows_per_rank // rows))
                    for rows in (2, 1)
                ]
                expected_points = (
                    sum(ordering_counts) if args.partition == "auto" else 2
                )
                assert len(shared) == expected_points, shared
                assert all(row["planning"]["mode"] == "symmetric" for row in shared)
                owners = [row["searched_by_rank"] for row in shared]
                expected_owners = (
                    set(range(min(world, max(ordering_counts))))
                    if args.partition == "auto"
                    else {0}
                )
                assert set(owners) == expected_owners, owners
                print(
                    json.dumps(
                        {
                            "rank": rank,
                            "event": "shared_sweep_prepared",
                            "points": len(shared),
                            "owners": owners,
                            "seconds": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
                errors, losses = [], []
                for step in range(3):
                    optimizer.zero_grad(set_to_none=True)
                    for peer in range(world):
                        objective(
                            reference,
                            data(peer, step),
                        ).backward()
                    optimizer.step()
                    measured = trainer.step(data(rank, step))
                    actual = snapshot(trainer)["model"]
                    for name, value in reference.named_parameters():
                        torch.testing.assert_close(
                            actual[name], value, rtol=2e-4, atol=2e-6
                        )
                        errors.append(
                            float((actual[name] - value.detach()).abs().max())
                        )
                    losses.append(measured.loss)
                result = {
                    "passed": True,
                    "rank": rank,
                    "world": world,
                    "rows_per_rank": args.rows_per_rank,
                    "points": len(shared),
                    "owners": owners,
                    "max_parameter_error": max(errors),
                    "losses": losses,
                }
                (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
                print(json.dumps(result), flush=True)
        finally:
            dist.destroy_process_group(group)
finally:
    dist.destroy_process_group()
