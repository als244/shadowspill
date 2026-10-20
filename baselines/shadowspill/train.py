"""One ShadowSpill case of a catalog model through the public training API,
``Trainer`` over the ``ShadowSpill`` backend; launch with torchrun.

    torchrun --standalone --nproc-per-node=N -m baselines.shadowspill.train \\
        --output DIR --model NAME --tokens-per-microbatch T \\
        --tokens-per-step S --execution-gib G --spill-gib H ...

The settings are ``baselines.shadowspill.settings``. Every rank writes
``rank-NNNNN/request.json`` as it starts, ``plan.json`` (and ``search.json``
when the backend searched) once planning is done, and ``result.json`` or
``failure.json`` as it ends; the first rank writes ``results.json`` with
every rank's measurements and ``row.json``, the result row every harness
shares.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime
import functools
import math
import os
import resource
import shutil
import sys
import tempfile
import time
import traceback
from fractions import Fraction
from pathlib import Path

import torch
import torch.distributed as dist

from baselines.common import summarize_cycles, write_json
from baselines.common.data import Corpus, synthetic_tokens
from baselines.common.models import useful_flops_per_token
from baselines.common.provenance import (
    environment_identity,
    hardware_identity,
    source_identity,
)
from baselines.common.reference import Access
from baselines.common.results import memory_peaks
from baselines.common.results import row as result_row
from baselines.common.training import TrainingModel
from baselines.shadowspill.settings import SOURCE_PATHS, add_arguments, from_namespace
from workloads.full_model import HEAD_LOSS_METRIC

GIB = 1 << 30


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, required=True)
    add_arguments(parser)
    return parser.parse_args(argv)


def optimizer_arguments(precision):
    """What mlops's AdamW is built with beyond its defaults, as the quickstart
    builds it: the dtype it reads the accumulated gradient at (the contract's
    gradient dtype, so a float32 sum is not rounded on its way in), the dtype
    of its moments (``parameter`` for the dtype of what it steps) and both
    roundings. A rounding the contract leaves unnamed is rounding to nearest,
    as every other harness's AdamW rounds: ShadowSpill's own training default
    rounds bfloat16 moments stochastically, which a case asks for by name."""

    given = {
        "gradient_dtype": getattr(torch, precision.grad_dtype or precision.model_dtype),
        "parameter_rounding": precision.parameter_rounding or "nearest",
        "opt_state_rounding": precision.opt_state_rounding or "nearest",
    }
    if precision.opt_state_dtype is not None:
        given["opt_state_dtype"] = (
            "parameter"
            if precision.opt_state_dtype == "parameter"
            else getattr(torch, precision.opt_state_dtype)
        )
    return given


def jsonable(value):
    """A report's values as JSON holds them: mappings and sequences by
    structure, exact fractions and anything else by their text."""

    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, bool | int | float | str) or value is None:
        return value
    if isinstance(value, Fraction):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        # Field by field: a copy of the whole would deep-copy whatever a
        # field refers to.
        return {
            field.name: jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    return repr(value)


def pool_record(statistics):
    """A pool's statistics as integers, by the runtime's own field names."""
    return {name: int(getattr(statistics, name)) for name, _ in statistics._fields_}


def plan_record(plan, planning_seconds):
    """What the admitted plan promises and what making it cost: the budgets
    it was planned against, its predicted step, peaks and traffic, the
    summary of its simulated step and the wall time of every phase."""

    return jsonable(
        {
            "planning_seconds": planning_seconds,
            "mode": plan.mode,
            "capture_identity": plan.capture_identity,
            "execution_budget_bytes": plan.execution_budget_bytes,
            "spill_budget_bytes": plan.spill_budget_bytes,
            "predicted_makespan_ns": plan.predicted_makespan_ns,
            "predicted_device_peak_bytes": plan.predicted_device_peak_bytes,
            "predicted_spill_peak_bytes": plan.predicted_spill_peak_bytes,
            "transfer_bytes_evicted": plan.transfer_bytes_evicted,
            "transfer_bytes_fetched": plan.transfer_bytes_fetched,
            "captured_stage_count": plan.captured_stage_count,
            "summary": plan.summary,
            "phase_timings_ns": dict(plan.phase_timings_ns),
            "data_ordering": plan.data_ordering,
            "optimizer_ordering": plan.optimizer_ordering,
        }
    )


def boundary(error):
    """Whether a failure is the frontier's boundary rather than a defect: the
    device ran out of memory, or ShadowSpill found no plan within the
    budgets it was given."""

    from shadowspill.errors import AdmissionError, PlanSearchExhaustedError

    if isinstance(
        error, torch.OutOfMemoryError | AdmissionError | PlanSearchExhaustedError
    ):
        return True
    return isinstance(error, RuntimeError) and "fits the configured budgets" in str(
        error
    )


class ShadowSpillAccess(Access):
    """Whole values through the public API: the stepped values (the masters
    where kept) and the moments from the checkpoint ``Trainer.save`` writes
    out of the pool, the buffers from the in-place state reader. Gradients
    are consumed by the step inside the pool and are not exposed; the
    moments the first update makes are the gradient's."""

    exposes = ("first_moment", "second_moment", "master")

    def __init__(self, trainer, runtime, directory, device):
        super().__init__(None, device)
        self.trainer = trainer
        self.runtime = runtime
        self.directory = Path(directory)
        self.count = 0
        named = list(trainer.model.named_parameters())
        self.names = {id(parameter): name for name, parameter in named}
        self.order = [name for name, _ in named]
        self.values = {}
        self.stepped = {}
        self.moments = {}

    def refresh(self):
        """Read the current state: a checkpoint written and read back, and
        the buffers in place."""

        from shadowspill.pytorch import read_model_state

        path = self.directory / f"snapshot-{self.count:04d}"
        self.count += 1
        self.trainer.save(path, weights="master")
        state = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
        shutil.rmtree(path)
        self.values = read_model_state(self.trainer.model, runtime=self.runtime)
        self.stepped = state["model"]
        optimizer = state["optimizer"]
        # The trainer builds its optimizer over every parameter of the model
        # in the model's order, so the entries are keyed by that position.
        indices = [
            index for group in optimizer["param_groups"] for index in group["params"]
        ]
        if len(indices) != len(self.order):
            raise ValueError("the optimizer's parameters are not the model's")
        self.moments = {}
        for name, index in zip(self.order, indices, strict=True):
            entry = optimizer["state"].get(index)
            if not entry:
                continue
            for key in ("exp_avg", "exp_avg_sq"):
                value = entry.get(key)
                if value is not None and tuple(value.shape) != tuple(
                    self.stepped[name].shape
                ):
                    raise ValueError(f"optimizer entry {index} is not {name}'s")
            self.moments[name] = entry

    def parameter(self, parameter):
        return self.stepped[self.names[id(parameter)]].to(self.device)

    def buffer(self, name, buffer):
        return self.values[name].to(self.device)

    def gradient(self, parameter):
        return None

    def state(self, parameter, key):
        entry = self.moments.get(self.names[id(parameter)])
        value = None if entry is None else entry.get(key)
        return None if value is None else value.to(self.device)

    def evidence(self, model):
        """The dtypes the stepped values and the moments are kept at, with
        the elements kept at each."""

        counts = {"stepped": {}, "moments": {}}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            stepped = self.stepped[name]
            key = str(stepped.dtype)
            counts["stepped"][key] = counts["stepped"].get(key, 0) + stepped.numel()
            for moment in ("exp_avg", "exp_avg_sq"):
                value = self.moments.get(name, {}).get(moment)
                if value is not None:
                    key = str(value.dtype)
                    counts["moments"][key] = (
                        counts["moments"].get(key, 0) + value.numel()
                    )
        if not counts["moments"]:
            raise ValueError("optimizer evidence is empty")
        return counts


def main(argv=None):
    args = parse(argv)
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    settings = from_namespace(args)
    geometry = settings.validate(world)
    request = settings.request()
    precision = settings.precision()
    torch.set_num_threads(settings.cpu_threads)
    # Nothing touches the device before the backend installs its allocator:
    # the device is named here and selected, probed and capped once the
    # runtime is up.
    device = torch.device("cuda", local_rank)
    started = time.perf_counter()
    output = args.output / f"rank-{rank:05d}"
    output.mkdir(parents=True, exist_ok=False)
    stage = "distributed_initialization"
    description = settings.describe(world)
    record = {
        **description,
        "output": str(args.output),
        "rank": rank,
        "world_size": world,
        "hardware": None,
        "resolved_precision": precision.resolved(),
        "microbatches_per_step": geometry.accumulation,
        "shadowspill": True,
    }
    write_json(output / "request.json", record)
    snapshots = None
    try:
        if world > 1:
            # The runtime takes a host control group before it allocates
            # its pools; the device group is made inside the backend.
            dist.init_process_group("gloo")
        stage = "model_initialization"
        from mlops.dispatch import set_weight_gradient_dtype

        grad_dtype = getattr(torch, precision.grad_dtype or precision.model_dtype)
        set_weight_gradient_dtype(grad_dtype)
        torch.manual_seed(settings.seed)
        # The model is built on meta; the backend materializes it in its
        # pools and runs the initialization there.
        model = TrainingModel(request, geometry, dtype=precision.model_dtype)
        useful = (
            useful_flops_per_token(model.model, geometry.sequence_length)
            * geometry.tokens_per_step
        )
        record["summary"] = request.summary(model.model)
        checkpoint = None
        if settings.checkpoint is not None:
            checkpoint = request.checkpoint(settings.checkpoint)
            record["checkpoint"] = {
                "directory": str(Path(settings.checkpoint).resolve()),
                "config": checkpoint.config.__dict__
                if not hasattr(checkpoint.config, "__dataclass_fields__")
                else {
                    name: getattr(checkpoint.config, name)
                    for name in checkpoint.config.__dataclass_fields__
                },
            }
        write_json(output / "request.json", record)
        loaded = {}

        def copy_value(parameter, value):
            parameter.copy_(value.to(parameter.dtype))

        def initialize(module):
            # Called by preparation on the pool-owned tensors: the model's
            # own initialization under the seed, then the checkpoint's
            # values written in place.
            from shadowspill.training import reset_parameters

            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(settings.seed)
                reset_parameters(module)
            if checkpoint is not None:
                with torch.no_grad():
                    loaded.update(checkpoint.load_into(module, copy=copy_value))

        vocabulary = int(model.model.config.vocab_size)
        corpus = None
        if settings.data is not None:
            corpus = Corpus(settings.data)
            corpus.check_vocabulary(vocabulary)
            record["corpus"] = corpus.meta
        tokens_per_rank = geometry.tokens_per_rank_per_step
        batch = geometry.tokens_per_microbatch
        microbatch_count = geometry.accumulation

        def item(step):
            if corpus is None:
                tokens, targets = synthetic_tokens(
                    vocabulary, tokens_per_rank, seed=settings.seed, rank=rank
                )
                return tokens, targets, 0
            return corpus.window(
                rank=rank, world_size=world, step=step, tokens=tokens_per_rank
            )

        def microbatches(data):
            # Each microbatch's objective is already its share of the
            # update, so every scale is one.
            tokens, targets = data
            for start in range(0, tokens.shape[1], batch):
                yield (
                    (
                        tokens[:, start : start + batch],
                        targets[:, start : start + batch],
                    ),
                    1.0,
                )

        def objective(module, data):
            # The backend sums the ranks' contributions, so each rank's
            # share of the update is divided by the world.
            loss, metrics = module(*data)
            return (loss / world if world > 1 else loss), metrics

        import mlops

        from shadowspill.pytorch import ProfilingOptions
        from shadowspill.training import Distributed, Trainer
        from shadowspill.training.backends import ShadowSpill
        from shadowspill.training.schedules import Constant

        # A case checked against the reference is never a throughput result,
        # so its tasks are timed as the numerical gate times them: the exact
        # task's warmups, the allocation probes and the minimum samples,
        # without the sustained-clock conditioning and duration floors a
        # throughput case's profiles take, which cost a tiny model's search
        # minutes.
        profiling = (
            ProfilingOptions(
                conditioning_seconds=0,
                conditioning_wall_seconds=0,
                measurement_seconds=0,
                measurement_wall_seconds=0,
            )
            if settings.verify_reference
            else None
        )
        record["profiling_options"] = (
            "defaults" if profiling is None else dataclasses.asdict(profiling)
        )
        stage = "runtime"
        with ShadowSpill(
            execution_gib=settings.execution_gib,
            spill_gib=settings.spill_gib,
            device=device,
            artifact_store=settings.artifact_store,
            profiling_options=profiling,
            round_accumulation_once=precision.round_accumulation_once,
            external_headroom_gib=settings.external_headroom_gib,
            control_group=dist.group.WORLD if world > 1 else None,
            host_headroom_gib=settings.host_headroom_gib,
        ) as backend:
            # The backend installed its allocator in PyTorch's place: the
            # device's peaks are the pool's statistics, and PyTorch's own
            # counters and cap do not exist under it.
            torch.cuda.set_device(device)
            record["hardware"] = hardware_identity()
            group = (
                dist.new_group(backend="nccl", device_id=backend.device)
                if world > 1
                else None
            )
            try:
                optimizer = functools.partial(
                    mlops.optim.AdamW, **optimizer_arguments(precision)
                )
                with Trainer(
                    model,
                    objective=objective,
                    optimizer=optimizer,
                    optimizer_args={
                        "betas": (0.9, 0.999),
                        "eps": 1e-8,
                        "weight_decay": settings.weight_decay,
                    },
                    schedules={"lr": Constant(settings.learning_rate)},
                    microbatches={"fixed": microbatches},
                    backend=backend,
                    master_dtype=None
                    if precision.master_dtype is None
                    else getattr(torch, precision.master_dtype),
                    grad_dtype=grad_dtype,
                    distributed=None if group is None else Distributed(group),
                ) as trainer:
                    stage = "planning"
                    tokens, targets, _ = item(0)
                    planning_started = time.perf_counter()
                    with (
                        checkpoint
                        if checkpoint is not None
                        else contextlib.nullcontext()
                    ):
                        trainer.prepare((tokens, targets), initialize=initialize)
                    torch.cuda.synchronize()
                    planning_seconds = time.perf_counter() - planning_started
                    if checkpoint is not None:
                        record["loaded"] = loaded
                        if loaded["unused"]:
                            raise ValueError(
                                "checkpoint keys the model did not take:"
                                f" {loaded['unused'][:5]}"
                            )
                    plan = trainer.plan
                    record["plan"] = plan_record(plan, planning_seconds)
                    (output / "plan.json").write_text(
                        plan.execution_plan.to_json() + "\n"
                    )
                    if trainer.planning is not None:
                        trainer.planning.save(output / "search.json")
                    record["pools_after_planning"] = {
                        name: pool_record(backend.runtime.pool_statistics(name))
                        for name in ("execution", "spill")
                    }
                    write_json(output / "request.json", record)
                    access = reference = None
                    if settings.verify_reference:
                        from baselines.common.reference import Reference

                        snapshots = tempfile.TemporaryDirectory(
                            prefix="shadowspill-baseline-"
                        )
                        access = ShadowSpillAccess(
                            trainer, backend.runtime, snapshots.name, device
                        )
                        access.refresh()
                        reference = Reference(
                            trainer.model,
                            request,
                            geometry,
                            precision,
                            settings,
                            device,
                            world,
                            access=access,
                        )
                    cycles, losses, head_losses, wrapped = [], [], [], 0
                    device_used = []
                    evidence = None
                    stage = "training"
                    setup_seconds = time.perf_counter() - started
                    print(
                        f"rank={rank} initialized seconds={setup_seconds:.3f} "
                        f"planning={planning_seconds:.3f} "
                        f"accumulation={microbatch_count}",
                        flush=True,
                    )
                    for step in range(settings.warmup_steps + settings.steps):
                        tokens, targets, wrapped = item(step)
                        expected_loss = (
                            None
                            if reference is None
                            else reference.step(
                                trainer.model,
                                tokens.to(device),
                                targets.to(device),
                                batch,
                            )
                        )
                        measured = step >= settings.warmup_steps
                        if step == settings.warmup_steps:
                            torch.cuda.synchronize()
                            if world > 1:
                                dist.barrier()
                            if settings.nsys_capture and rank == 0:
                                torch.cuda.cudart().cudaProfilerStart()
                        torch.cuda.synchronize()
                        if world > 1:
                            dist.barrier()
                        tick = time.perf_counter()
                        torch.cuda.nvtx.range_push(
                            f"baseline.shadowspill/rank-{rank}/step-{step + 1}"
                        )
                        result = trainer.step((tokens, targets))
                        torch.cuda.nvtx.range_pop()
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter() - tick
                        # The rank's share of the update, as every harness
                        # records it; the objective divided it by the world.
                        loss_sum = float(sum(result.losses)) * world
                        head_sum = 0.0
                        for loss, metrics in zip(
                            result.losses, result.metrics, strict=True
                        ):
                            head = (
                                metrics.get(HEAD_LOSS_METRIC)
                                if isinstance(metrics, dict)
                                else None
                            )
                            head_sum += (
                                float(loss) * world if head is None else float(head)
                            )
                        if not math.isfinite(loss_sum):
                            raise ValueError("training loss is not finite")
                        free, total = torch.cuda.mem_get_info(device)
                        device_used.append(total - free)
                        if reference is not None:
                            access.refresh()
                            reference.check(
                                trainer.model,
                                None,
                                torch.tensor(loss_sum, device=device),
                                expected_loss,
                            )
                        if measured:
                            cycles.append(elapsed)
                            losses.append(loss_sum)
                            head_losses.append(head_sum)
                        if step == 0 and access is not None:
                            evidence = access.evidence(trainer.model)
                        print(
                            f"rank={rank} step={step + 1} measured={measured} "
                            f"seconds={elapsed:.6f} loss={loss_sum:.6f}",
                            flush=True,
                        )
                    if settings.nsys_capture:
                        if world > 1:
                            dist.barrier()
                        if rank == 0:
                            torch.cuda.cudart().cudaProfilerStop()
                    torch.cuda.synchronize()
                    execution = backend.runtime.pool_statistics("execution")
                    spill = backend.runtime.pool_statistics("spill")
                    performance = (
                        reference is None
                        and not settings.nsys_capture
                        and request.entry.size not in ("tiny", "numerical")
                    )
                    result = {
                        "status": "completed",
                        "rank": rank,
                        "cycles": cycles,
                        "losses": losses,
                        "head_losses": head_losses,
                        "precision_evidence": evidence,
                        "corpus_wrapped": wrapped,
                        # The device peak is the execution pool's measured
                        # peak occupancy; what the process held is the
                        # budget, the pool's physical capacity, whole from
                        # the first second.
                        "peak_allocated_bytes": int(execution.peak_allocated_bytes),
                        "peak_reserved_bytes": int(settings.execution_gib * GIB),
                        "host_peak_rss_bytes": resource.getrusage(
                            resource.RUSAGE_SELF
                        ).ru_maxrss
                        * 1024,
                        "shadowspill_memory": {
                            "execution_pool": pool_record(execution),
                            "spill_pool": pool_record(spill),
                            "predicted_device_peak_bytes": int(
                                plan.predicted_device_peak_bytes
                            ),
                            "predicted_spill_peak_bytes": int(
                                plan.predicted_spill_peak_bytes
                            ),
                            "device_used_bytes_max": int(max(device_used)),
                        },
                        "planning_seconds": planning_seconds,
                        "setup_seconds": setup_seconds,
                        "total_wall_seconds": time.perf_counter() - started,
                        "finished_at": datetime.datetime.now().astimezone().isoformat(),
                        "qualified_against_reference": reference is not None,
                        "reference_checks": None
                        if reference is None
                        else reference.checks,
                        "performance_result": performance,
                    }
                    write_json(output / "result.json", result)
                    all_ranks = [result]
                    if world > 1:
                        all_ranks = [None] * world
                        dist.all_gather_object(all_ranks, result)
                    if rank == 0:
                        summary = summarize_cycles(
                            [row["cycles"] for row in all_ranks],
                            global_tokens=geometry.tokens_per_step,
                            useful_flops_per_step=useful,
                        )
                        summary["performance_result"] = performance
                        summary["losses"] = [row["losses"] for row in all_ranks]
                        summary["head_losses"] = [
                            row["head_losses"] for row in all_ranks
                        ]
                        write_json(
                            args.output / "results.json",
                            {
                                **summary,
                                "status": "completed",
                                "request": record,
                                "ranks": all_ranks,
                                "useful_flops_per_step": useful,
                                "flops_convention": "baselines.common.models"
                                ".useful_flops_per_token",
                            },
                        )
                        memory = memory_peaks(all_ranks)
                        memory["peak_allocated_aggregation"] = (
                            "maximum over ranks of the execution pool's measured"
                            " peak occupancy"
                        )
                        memory["peak_reserved_aggregation"] = (
                            "the execution budget, the pool's physical capacity"
                        )
                        memory["shadowspill"] = [
                            row["shadowspill_memory"] for row in all_ranks
                        ]
                        write_json(
                            args.output / "row.json",
                            result_row(
                                harness="shadowspill",
                                status="completed",
                                **description,
                                hardware=record["hardware"],
                                provenance={
                                    "source": source_identity(SOURCE_PATHS),
                                    "environment": environment_identity(sys.executable),
                                    "model_summary": record["summary"],
                                    "corpus": record.get("corpus"),
                                    "checkpoint": record.get("checkpoint"),
                                    "plan": record["plan"],
                                    "pools_after_planning": record[
                                        "pools_after_planning"
                                    ],
                                },
                                measurement=summary,
                                memory=memory,
                            ),
                        )
            finally:
                if group is not None:
                    dist.destroy_process_group(group)
    except Exception as error:
        write_json(
            output / "failure.json",
            {
                "rank": rank,
                "stage": stage,
                # A budget the planner cannot fit is the frontier's boundary,
                # recorded under the schema's name for it.
                "status": "cuda_out_of_memory" if boundary(error) else "failed",
                "boundary": boundary(error),
                "exception_type": type(error).__name__,
                "error": traceback.format_exc(),
            },
        )
        raise
    finally:
        if snapshots is not None:
            snapshots.cleanup()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
