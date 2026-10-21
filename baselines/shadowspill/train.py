"""One ShadowSpill case of a catalog model through the public training API,
``Trainer`` over the ``ShadowSpill`` backend: the trainer
``baselines.common.launch`` starts under torchrun.

The settings are ``baselines.shadowspill.settings``, validated by the
launcher; ``baselines.common.runner`` writes the request, the result and
the row around ``train``, which plans the step and runs every step under the
runner's clock, writing ``plan.json`` (and ``search.json`` when the backend
searched) beside the request once planning is done.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import shutil
import tempfile
import time
from fractions import Fraction
from pathlib import Path

import torch
import torch.distributed as dist

from baselines.common import runner
from baselines.common.provenance import hardware_identity
from baselines.common.reference import Access
from baselines.common.results import memory_peaks
from baselines.common.training import TrainingModel
from baselines.shadowspill import settings as engine_settings
from workloads.full_model import HEAD_LOSS_METRIC

GIB = 1 << 30


def optimizer_arguments(precision):
    """What mlops's AdamW is built with beyond its defaults, as the quickstart
    builds it: the dtype it reads the accumulated gradient at (the contract's
    gradient dtype, so a float32 sum is not rounded on its way in), the dtype
    of its moments (``parameter`` for the dtype of what it steps) and both
    roundings. A rounding the contract leaves unnamed is rounding to nearest,
    as every other engine's AdamW rounds: ShadowSpill's own training default
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


def failure(error):
    """A budget the planner cannot fit is the frontier's boundary, recorded
    under the schema's name for it."""

    found = boundary(error)
    return {"status": "cuda_out_of_memory" if found else "failed", "boundary": found}


def memory(all_ranks):
    """The row's memory: the execution pool's peaks and budget, with what
    each rank's pools held."""

    value = memory_peaks(all_ranks)
    value["peak_allocated_aggregation"] = (
        "maximum over ranks of the execution pool's measured peak occupancy"
    )
    value["peak_reserved_aggregation"] = (
        "the execution budget, the pool's physical capacity"
    )
    value["shadowspill"] = [row["shadowspill_memory"] for row in all_ranks]
    return value


def train(case, clock):
    """Plan the case's step with the ShadowSpill backend and run every step."""

    settings, geometry, precision = case.settings, case.geometry, case.precision
    request = settings.request()
    record = case.record
    world, device = case.world, case.device
    if world > 1:
        # The runtime takes a host control group before it allocates its
        # pools; the device group is made inside the backend.
        dist.init_process_group("gloo")
    case.stage = "model_initialization"
    from mlops.dispatch import set_weight_gradient_dtype

    grad_dtype = getattr(torch, precision.grad_dtype or precision.model_dtype)
    set_weight_gradient_dtype(grad_dtype)
    torch.manual_seed(settings.seed)
    # The model is built on meta; the backend materializes it in its pools
    # and runs the initialization there.
    model = TrainingModel(request, geometry, dtype=precision.model_dtype)
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
    case.save()
    loaded = {}

    def copy_value(parameter, value):
        parameter.copy_(value.to(parameter.dtype))

    def initialize(module):
        # Called by preparation on the pool-owned tensors: the model's own
        # initialization under the seed, then the checkpoint's values
        # written in place.
        from shadowspill.training import reset_parameters

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(settings.seed)
            reset_parameters(module)
        if checkpoint is not None:
            with torch.no_grad():
                loaded.update(module.load_checkpoint(checkpoint, copy=copy_value))

    data = case.data(int(model.model.config.vocab_size))
    batch = geometry.tokens_per_microbatch

    def microbatches(pair):
        # Each microbatch's objective is already its share of the update,
        # so every scale is one.
        tokens, targets = pair
        for start in range(0, tokens.shape[1], batch):
            yield (
                (tokens[:, start : start + batch], targets[:, start : start + batch]),
                1.0,
            )

    def objective(module, pair):
        # The backend sums the ranks' contributions, so each rank's share
        # of the update is divided by the world.
        loss, metrics = module(*pair)
        return (loss / world if world > 1 else loss), metrics

    import mlops

    from shadowspill.planner import SearchOptions
    from shadowspill.planner.search.algorithms.pressurefit import PressureFit
    from shadowspill.planner.search.algorithms.pressurefit.options import (
        PressureFitOptions,
    )
    from shadowspill.planner.search.toolkit.resolution import (
        named_resolution_options,
    )
    from shadowspill.pytorch import ProfilingOptions
    from shadowspill.training import Distributed, Trainer
    from shadowspill.training.backends import ShadowSpill
    from shadowspill.training.schedules import Constant

    # A case checked against the reference is never a throughput result, so
    # its tasks are timed as the numerical gate times them: the exact task's
    # warmups, the allocation probes and the minimum samples, without the
    # sustained-clock conditioning and duration floors a throughput case's
    # profiles take, which cost a tiny model's search minutes.
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
    shares = named_resolution_options(settings.resolution_options)
    record["resolution_options"] = list(shares)
    search = SearchOptions(
        algorithm=PressureFit(
            PressureFitOptions(
                resolution_options=tuple(Fraction(share) for share in shares)
            )
        )
    )
    snapshots = None
    case.stage = "runtime"
    try:
        with ShadowSpill(
            execution_gib=settings.execution_gib,
            spill_gib=settings.spill_gib,
            device=device,
            artifact_store=settings.artifact_store,
            search_options=search,
            profiling_options=profiling,
            fuse_narrow_gradient_accumulation=precision.fuse_narrow_gradient_accumulation,
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
                    case.stage = "planning"
                    planning_started = time.perf_counter()
                    with (
                        checkpoint
                        if checkpoint is not None
                        else contextlib.nullcontext()
                    ):
                        trainer.prepare(data(0), initialize=initialize)
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
                    (case.directory / "plan.json").write_text(
                        plan.execution_plan.to_json() + "\n"
                    )
                    if trainer.planning is not None:
                        trainer.planning.save(case.directory / "search.json")
                    record["pools_after_planning"] = {
                        name: pool_record(backend.runtime.pool_statistics(name))
                        for name in ("execution", "spill")
                    }
                    case.save()
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
                    device_used = []
                    evidence = None
                    clock.ready(f"planning={planning_seconds:.3f}")
                    for step in range(clock.steps):
                        tokens, targets = data(step)
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
                        clock.begin(step)
                        result = trainer.step((tokens, targets))
                        # The rank's share of the update, as every engine
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
                        clock.end(step, loss_sum, head_sum)
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
                        if step == 0 and access is not None:
                            evidence = access.evidence(trainer.model)
                    clock.close()
                    execution = backend.runtime.pool_statistics("execution")
                    spill = backend.runtime.pool_statistics("spill")
            finally:
                if group is not None:
                    dist.destroy_process_group(group)
    finally:
        if snapshots is not None:
            snapshots.cleanup()
    return runner.Outcome(
        evidence=evidence,
        reference_checks=None if reference is None else reference.checks,
        # The device peak is the execution pool's measured peak occupancy;
        # what the process held is the budget, the pool's physical
        # capacity, whole from the first second.
        peaks={
            "peak_allocated_bytes": int(execution.peak_allocated_bytes),
            "peak_reserved_bytes": int(settings.execution_gib * GIB),
        },
        result={
            "shadowspill_memory": {
                "execution_pool": pool_record(execution),
                "spill_pool": pool_record(spill),
                "predicted_device_peak_bytes": int(plan.predicted_device_peak_bytes),
                "predicted_spill_peak_bytes": int(plan.predicted_spill_peak_bytes),
                "device_used_bytes_max": int(max(device_used)),
            },
            "planning_seconds": planning_seconds,
        },
        provenance={
            "plan": record["plan"],
            "pools_after_planning": record["pools_after_planning"],
        },
        memory=memory,
    )


def main(argv=None):
    runner.main(
        "shadowspill",
        engine_settings,
        train,
        argv,
        standalone=False,
        device_first=False,
        failure=failure,
    )


if __name__ == "__main__":
    main()
