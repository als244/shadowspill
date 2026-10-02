"""Real, separate-device DP correctness using the public Trainer/Runtime path.

Launch with torchrun. Results are checked against a combined-data CPU oracle.
Any forced graph-pair selection is a test-only search restriction; capture,
profiling, admission, execution, and collectives use production code unchanged.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import math
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn

from shadowspill.pytorch import ProfilingOptions
from shadowspill.training import Distributed, Trainer
from shadowspill.training.backends import ShadowSpill
from shadowspill.training.observations import parameter_norms


class Model(nn.Module):
    def __init__(self, dtype=torch.float32, *, seed=17, device="cpu"):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.first = nn.Linear(7, 9, dtype=dtype, device=device)
            self.second = nn.Linear(9, 5, dtype=dtype, device=device)
        self.register_buffer("local_counter", torch.tensor(seed, device=device))

    def forward(self, x):
        self.local_counter.add_(1)
        return self.second(torch.tanh(self.first(x)))


def objective(model, data):
    x, y, denominator = data
    return (model(x).float() - y.float()).square().sum() / denominator


def batch(rank, step, dtype, world):
    generator = torch.Generator().manual_seed(1234 + rank * 101 + step)
    # Unequal batches ensure SUM/global normalization, rather than mean-of-means.
    rows = 4 + 2 * rank
    denominator = sum(4 + 2 * other for other in range(world)) * 5
    return (
        torch.randn(rows, 7, generator=generator).to(dtype),
        torch.randn(rows, 5, generator=generator).to(dtype),
        float(denominator),
    )


def snapshot(trainer, *, weights="compute"):
    return trainer._execution.call.state_dict(weights=weights)


def equal_tree(actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            equal_tree(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            equal_tree(a, b)
    else:
        assert actual == expected, (actual, expected)


def owned(value, rank, world, sharded):
    if not sharded:
        return value
    capacity = math.ceil(value.numel() / world)
    padded = torch.zeros(capacity * world, dtype=value.dtype)
    padded[: value.numel()].copy_(value.reshape(-1))
    return padded.narrow(0, rank * capacity, capacity)


def restrict_variant(variant):
    if variant == "auto":
        return
    from shadowspill.ir import TaskAlternativeChoice
    from shadowspill.pytorch.distributed import _selection

    def choices(program, resolution_options):
        selections = []
        for group in program.task_alternative_groups:
            available = {item.option_id for item in group.options}
            assert variant in available, (variant, available)
            selections.append(TaskAlternativeChoice(group.group_id, variant))
        return (tuple(selections),)

    _selection.resolutions = choices


def run(args):
    faulthandler.enable()
    faulthandler.dump_traceback_later(600, repeat=True)
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    assert world >= 2, "This test requires separate devices and processes."
    output = args.out.resolve()
    rank_dir = output / f"rank-{rank:05d}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float16 if args.precision == "fp16" else torch.float32
    restrict_variant(args.variant)
    started = time.monotonic()

    def log(event, **details):
        record = dict(
            time=datetime.now(UTC).isoformat(),
            rank=rank,
            event=event,
            elapsed_s=time.monotonic() - started,
            **details,
        )
        print(json.dumps(record), flush=True)
        with (rank_dir / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")

    # FP32 oracle, initialized from exactly the selected compute representation.
    reference = Model(dtype).float()
    from ._optimizers import MatrixNormalizedMomentum

    reference_constructor = (
        MatrixNormalizedMomentum if args.optimizer == "matrix" else torch.optim.AdamW
    )
    reference_optimizer = reference_constructor(
        reference.parameters(),
        lr=0.005,
        betas=(0.8, 0.95),
        eps=1e-4,
        weight_decay=0.0,
        foreach=False,
    )
    dist.init_process_group("gloo", timeout=timedelta(seconds=300))
    try:
        log(
            "begin",
            precision=args.precision,
            masters=args.masters,
            sharded=args.sharded,
            optimizer=args.optimizer,
            variant=args.variant,
            world=world,
            local_rank=local_rank,
        )
        with ShadowSpill(
            device="auto",
            execution_gib=2,
            spill_gib=2,
            control_group=dist.group.WORLD,
            preparation_timeout=300,
            artifact_store=rank_dir / "artifacts",
            partition="whole",
            profiling_options=ProfilingOptions(
                conditioning_seconds=0,
                measurement_seconds=0,
                minimum_samples=3,
            ),
        ) as backend:
            assert backend.device.index == local_rank
            gpu = torch.cuda.get_device_properties(backend.device)
            log(
                "runtime_ready",
                device=str(backend.device),
                name=gpu.name,
                uuid=str(gpu.uuid),
            )
            group = dist.new_group(
                backend="nccl", device_id=backend.device, timeout=timedelta(seconds=300)
            )
            try:
                optimizer = torch.optim.AdamW
                optimizer_args = dict(
                    lr=0.005,
                    betas=(0.8, 0.95),
                    eps=1e-4,
                    weight_decay=0.0,
                    foreach=False,
                )
                if args.optimizer == "mlops":
                    from mlops.optim import AdamW

                    optimizer = AdamW
                    optimizer_args.update(
                        gradient_dtype=torch.float32, opt_state_dtype=torch.float32
                    )
                    if args.stochastic:
                        optimizer_args["parameter_rounding"] = "stochastic"

                if args.optimizer == "matrix":
                    assert not args.sharded
                    optimizer = MatrixNormalizedMomentum

                def make_trainer(model):
                    return Trainer(
                        model,
                        objective=objective,
                        optimizer=optimizer,
                        optimizer_args=optimizer_args,
                        backend=backend,
                        distributed=Distributed(group, timeout=300),
                        grad_dtype=torch.float32,
                        master_dtype=torch.float32 if args.masters else None,
                        shard_optimizer=args.sharded,
                        parameter_metrics=parameter_norms
                        if args.parameter_metrics
                        else None,
                    )

                example = batch(rank, 0, dtype, world)
                # Replica initialization updates the imported model, never its
                # source. Writing even identical bytes would privatize a mapped
                # checkpoint and add a full host model copy on every rank.
                source_model = Model(dtype, seed=17 + rank)
                source_values = {
                    name: (value.detach().clone(), value._version, value.data_ptr())
                    for name, value in source_model.named_parameters()
                }
                with make_trainer(source_model) as trainer:
                    log("prepare_start")
                    trainer.prepare(example)
                    for name, value in source_model.named_parameters():
                        original, version, pointer = source_values[name]
                        torch.testing.assert_close(value, original, rtol=0, atol=0)
                        assert value._version == version
                        assert value.data_ptr() == pointer
                    report = trainer.plan
                    assert report.execution_device == local_rank
                    assert report.program.devices[0].index == local_rank
                    selected_variants = [
                        item.option_id for item in report.search_result.selections
                    ]
                    if args.variant != "auto":
                        assert selected_variants and set(selected_variants) == {
                            args.variant
                        }
                    (rank_dir / "plan.json").write_text(
                        json.dumps(report.execution_plan.to_dict(), indent=2) + "\n"
                    )
                    log(
                        "prepared",
                        selected=trainer.selected_candidate,
                        variants=selected_variants,
                    )
                    diagnostic_tasks = None
                    if args.diagnostics:
                        from shadowspill.pytorch.execution import training as dispatch

                        diagnostic_tasks = []
                        execute_task = dispatch.execute_task

                        def counted_task(executor, run, record):
                            diagnostic_tasks.append(record.task.task_id)
                            return execute_task(executor, run, record)

                        dispatch.execute_task = counted_task
                        execution = trainer._execution.call._executor.run.execution
                        expected_tasks = [record.task.task_id for record in execution]
                        assert args.optimizer == "mlops"
                        initial_states = {
                            mode: snapshot(trainer, weights=mode)
                            for mode in ("master", "compute")
                        }
                        cpu_rng = torch.get_rng_state()
                        device_rng = torch.cuda.get_rng_state(backend.device)
                        diagnostic = trainer.diagnose(
                            example, directory=rank_dir / "startup"
                        )
                        assert diagnostic["buffer_snapshot_bytes"] == 8
                        assert diagnostic_tasks == expected_tasks * 2
                        diagnostic_tasks.clear()
                        assert trainer.step_count == 0
                        for mode, before in initial_states.items():
                            equal_tree(snapshot(trainer, weights=mode), before)
                        assert torch.equal(torch.get_rng_state(), cpu_rng)
                        assert torch.equal(
                            torch.cuda.get_rng_state(backend.device), device_rng
                        )
                        assert (rank_dir / "startup/timelines/traced.html").is_file()
                        assert (rank_dir / "startup/timelines/simulated.html").is_file()
                        log(
                            "startup_diagnostics_passed",
                            **diagnostic,
                            task_ids=expected_tasks,
                            identical_task_sequence=True,
                        )
                    initial = snapshot(trainer)
                    for name, parameter in reference.named_parameters():
                        torch.testing.assert_close(
                            initial["model"][name].float(), parameter, rtol=0, atol=0
                        )
                    assert initial["model"]["local_counter"].item() == 17 + rank
                    losses, worst_error = [], 0.0
                    for step in range(3):
                        reference_optimizer.zero_grad(set_to_none=True)
                        expected_losses = []
                        for peer in range(world):
                            x, y, denominator = batch(peer, step, dtype, world)
                            loss = objective(
                                reference, (x.float(), y.float(), denominator)
                            )
                            expected_losses.append(float(loss.detach()))
                            loss.backward()
                        reference_optimizer.step()
                        result = trainer.step(batch(rank, step, dtype, world))
                        if args.parameter_metrics:
                            assert result.parameter_metrics
                            assert all(
                                value.device.type == "cpu"
                                and not value.is_pinned()
                                and torch.isfinite(value).all()
                                for metrics in result.parameter_metrics.values()
                                for value in metrics.values()
                            )
                        if diagnostic_tasks is not None:
                            assert diagnostic_tasks == expected_tasks
                            diagnostic_tasks.clear()
                        actual = snapshot(trainer)
                        replicas = [None] * world
                        dist.all_gather_object(replicas, actual["model"])
                        for other in replicas:
                            for name in dict(reference.named_parameters()):
                                torch.testing.assert_close(
                                    actual["model"][name], other[name], rtol=0, atol=0
                                )
                        rtol, atol = (
                            (2e-4, 2e-6) if dtype == torch.float32 else (0.02, 0.0015)
                        )
                        for index, (name, parameter) in enumerate(
                            reference.named_parameters()
                        ):
                            value = actual["model"][name].float()
                            error = float((value - parameter.detach()).abs().max())
                            worst_error = max(worst_error, error)
                            torch.testing.assert_close(
                                value, parameter, rtol=rtol, atol=atol
                            )
                            expected_state = reference_optimizer.state[parameter]
                            found_state = actual["optimizer"]["state"][index]
                            state_keys = (
                                ("velocity", "column_energy")
                                if args.optimizer == "matrix"
                                else ("exp_avg", "exp_avg_sq")
                            )
                            for key in state_keys:
                                expected = owned(
                                    expected_state[key], rank, world, args.sharded
                                )
                                torch.testing.assert_close(
                                    found_state[key], expected, rtol=rtol, atol=atol
                                )
                        torch.testing.assert_close(
                            torch.tensor(result.loss),
                            torch.tensor(expected_losses[rank]),
                            rtol=rtol,
                            atol=atol,
                        )
                        losses.append(result.loss)
                        log(
                            "step_checked",
                            step=step + 1,
                            local_loss=result.loss,
                            max_parameter_error=worst_error,
                            seconds=result.seconds,
                        )

                    # Master-only and compute-only checkpoints each replay exactly
                    # relative to their own restored representation.
                    for weights in ("master", "compute"):
                        checkpoint = output / f"checkpoint-{weights}"
                        trainer.save(checkpoint, weights=weights)
                        record = torch.load(
                            checkpoint / f"rank-{rank:05d}" / "state.pt",
                            weights_only=True,
                            map_location="cpu",
                        )
                        assert not set(record["masters"]).intersection(record["model"])
                        assert bool(record["masters"]) == (
                            args.masters and weights == "master"
                        )
                        trainer.step(example)
                        trainer.load(checkpoint)
                        trainer.step(example)
                        replay = snapshot(trainer, weights="master")
                        trainer.load(checkpoint)
                        trainer.step(example)
                        equal_tree(snapshot(trainer, weights="master"), replay)
                        log("checkpoint_replay_checked", weights=weights)
                    fresh_checkpoint = checkpoint
                    expected_fresh = replay

                with make_trainer(Model(dtype, device="meta")) as restored:
                    log("fresh_restore_start")
                    restored.prepare(example, checkpoint=fresh_checkpoint)
                    restored.step(example)
                    equal_tree(snapshot(restored, weights="master"), expected_fresh)
                    evaluation = restored.evaluate([example])
                    assert math.isfinite(evaluation.mean_loss)
                    log("fresh_restore_and_forward_checked", loss=evaluation.mean_loss)

                record = dict(
                    passed=True,
                    rank=rank,
                    world=world,
                    precision=args.precision,
                    masters=args.masters,
                    shard_optimizer=args.sharded,
                    optimizer=args.optimizer,
                    variant=args.variant,
                    stochastic=args.stochastic,
                    losses=losses,
                    max_parameter_error=worst_error,
                    checkpoint_policies=["master", "compute"],
                    fresh_meta_restore=True,
                    forward_evaluation=True,
                )
                (rank_dir / "result.json").write_text(
                    json.dumps(record, indent=2) + "\n"
                )
                records = [None] * world
                dist.all_gather_object(records, record)
                if rank == 0:
                    (output / "result.json").write_text(
                        json.dumps(records, indent=2) + "\n"
                    )
                log("passed")
            finally:
                dist.destroy_process_group(group)
    finally:
        dist.destroy_process_group()
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--masters", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--parameter-metrics", action="store_true")
    parser.add_argument(
        "--optimizer", choices=("torch", "mlops", "matrix"), default="torch"
    )
    parser.add_argument(
        "--sharded", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--variant", choices=("auto", "save", "recompute"), default="auto"
    )
    run(parser.parse_args())
