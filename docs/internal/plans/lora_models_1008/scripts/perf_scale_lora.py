"""Full training versus LoRA at the unchanged performance-gate model sizes.

Use a fresh process per case. Both modes use the gate's geometry, budgets,
objective, BF16 base/gradient/moment precision and timing protocol. LoRA keeps
its default FP32 factor storage. No model checkpoints are copied or retained.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from collections.abc import Mapping
from dataclasses import asdict, fields, replace
import gc
import json
import math
from pathlib import Path
import resource
import statistics
import sys
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "workloads").is_dir())
sys.path.insert(0, str(ROOT))

import torch
from mlops.dispatch import set_weight_gradient_dtype
from qualification.model_state import import_case_model, release_case_model
from qualification.performance.cases import manifest_for
from qualification.performance.phases import _calibrated_runtime, _measure_groups
from qualification.performance.readings import _profile_metadata, _runtime_delta, _wait_idle
from qualification.runtime_evidence import adapter_statistics, check_physical_budget
from shadowspill.ir import TaskAlternativeChoice
from shadowspill.pytorch import plan_step
from workloads.full_model import build_case
from workloads.lora import configure_lora, parameter_report

from full_model_lora import pool_stats, save_pairs

GIB = 1 << 30


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=str) + "\n")
    temporary.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--family", choices=("llama3", "qwen35", "olmoe"), default="llama3")
    parser.add_argument("--mode", choices=("full", "lora"), default="full")
    parser.add_argument("--variant", choices=("save", "recompute", "auto"), default="save")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--factor-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--execution-gib", type=int, default=16)
    parser.add_argument("--spill-gib", type=int, default=112)
    parser.add_argument("--groups", type=int, default=3)
    parser.add_argument("--steps-per-group", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20_260_811)
    parser.add_argument("--outdir", type=Path)
    known, _ = parser.parse_known_args()
    if known.config:
        parser.set_defaults(**json.loads(known.config.read_text()))
    args = parser.parse_args()
    if args.outdir is None:
        parser.error("--outdir is required")
    args.outdir = Path(args.outdir)
    if min(args.groups, args.steps_per_group, args.warmup, args.rank,
           args.execution_gib, args.spill_gib) <= 0:
        parser.error("counts and budgets must be positive")
    return args


def optimizer_audit(outdir, inventory):
    programs = list((outdir / "artifacts/v1/planning/programs").glob("*/*/program.json"))
    if len(programs) != 1:
        raise AssertionError(f"expected one saved program, got {programs}")
    program = json.loads(programs[0].read_text())
    objects = {obj["object_id"]: obj for obj in program["objects"]}
    optimizer_inputs = {oid for task in program["tasks"] if task["phase"] == "optimizer"
                        for oid in task["inputs"]}
    observed = {role: sum(objects[oid]["size_bytes"] for oid in optimizer_inputs
                          if objects[oid]["role"] == role)
                for role in ("parameter", "gradient", "optimizer_state")}
    trainable = inventory["trainable_parameters"]
    expected = dict(parameter=sum(p["bytes"] for p in inventory["trainable"]),
                    gradient=2 * trainable,
                    optimizer_state=4 * trainable + 8 * len(inventory["trainable"]))
    audit = dict(observed=observed, expected=expected, status="passed" if observed == expected else "failed",
                 gradient_dtype="bfloat16", moment_dtype="bfloat16", source_program=str(programs[0]))
    write_json(outdir / "optimizer-audit.json", audit)
    if observed != expected:
        raise AssertionError(audit)


def run(args):
    torch.set_num_threads(4)
    set_weight_gradient_dtype(torch.bfloat16)
    outdir = args.outdir
    manifest = replace(manifest_for(args.family, "mlops"),
                       device_physical_capacity_bytes=args.execution_gib * GIB,
                       spill_budget_bytes=args.spill_gib * GIB)
    settings = {**vars(args), "outdir": str(outdir), "config": str(args.config) if args.config else None,
                "manifest": manifest.as_dict(), "head": "frozen" if args.mode == "lora" else "full"}
    write_json(outdir / "config.json", settings)
    print("START", json.dumps(settings), flush=True)
    runtime, capabilities, calibration_attempts = _calibrated_runtime(manifest)
    settings["device_name"] = torch.cuda.get_device_name()
    write_json(outdir / "config.json", settings)
    case = build_case(manifest, seed=args.seed)
    # Construct identical frozen base values and identical inputs before LoRA
    # adds factors and consumes additional random values.
    if args.mode == "lora":
        configure_lora(case.model, rank=args.rank, alpha=args.rank, factor_dtype=args.factor_dtype)
    inventory = parameter_report(case.model)
    inventory["model_config"] = asdict(case.model.config)
    write_json(outdir / "parameters.json", inventory)
    print("PARAMETERS", inventory["total_parameters"], "TRAINABLE", inventory["trainable_parameters"], flush=True)
    case = import_case_model(case, runtime=runtime)
    gc.collect()

    def choices(program, _shares):
        return (tuple(TaskAlternativeChoice(group.group_id, args.variant)
                      for group in program.task_alternative_groups),)

    selection = patch("shadowspill.planner.search.algorithms.pressurefit.resolutions", choices) if args.variant != "auto" else nullcontext()
    started = time.perf_counter()
    with selection:
        training = plan_step(case.model, objective=case.objective, optimizer=case.optimizer,
                             **manifest.dtypes.plan_arguments(), hyperparams=("lr",),
                             example_inputs=case.microbatches, runtime=runtime,
                             execution="execution", spill="spill", spill_budget=manifest.spill_budget_bytes,
                             optimizer_ordering="stage_interleaved", verbose=True,
                             artifact_store=outdir / "artifacts",
                             profiling_metadata=_profile_metadata(case.microbatches))
    preparation_seconds = time.perf_counter() - started
    save_pairs(training, outdir)
    optimizer_audit(outdir, inventory)
    report = training.plan_report
    write_json(outdir / "selections.json", [str(s) for s in report.execution_plan.selections])
    write_json(outdir / "plan-summary.json",
               {field.name: dict(value) if isinstance(value := getattr(report.summary, field.name), Mapping) else value
                for field in fields(report.summary)})
    stats_context = SimpleNamespace(runtime=runtime)
    prepared_pools = pool_stats(stats_context)
    prepared_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    print("PREPARED", preparation_seconds, "seconds; predicted step", report.predicted_makespan_ns / 1e9,
          "seconds; peak host GiB", prepared_rss/GIB, flush=True)

    # Exact full accumulated updates, including AdamW kernels, with no changes
    # to weights/moments/step counters. This also reaches sustained GPU clocks.
    for warm in range(args.warmup):
        started = time.perf_counter()
        result = training(case.microbatches, hyperparams={"lr": 0.0})
        training.mark_cycle_end()
        _wait_idle(training)
        training.invocation_timings()
        loss = sum(float(value) for value in result.objectives)
        if not math.isfinite(loss):
            raise AssertionError(f"nonfinite warmup loss: {loss}")
        del result
        print("WARMUP", warm + 1, "seconds", time.perf_counter()-started, "loss", loss, flush=True)
    gc.collect()
    torch.cuda.synchronize()
    physical_statuses = [check_physical_budget()]
    runtime_before = adapter_statistics()
    measured = _measure_groups(training, case, manifest,
                               SimpleNamespace(groups=args.groups, steps_per_group=args.steps_per_group,
                                               profiler_annotations=False), physical_statuses)
    losses = [sum(values) for values in measured.measured_objectives]
    if not all(math.isfinite(loss) for loss in losses):
        raise AssertionError(f"nonfinite measured losses: {losses}")
    seconds = statistics.median(measured.group_seconds) / args.steps_per_group
    summary = dict(status="passed", config=settings, parameters=inventory["total_parameters"],
                   trainable=inventory["trainable_parameters"], preparation_seconds=preparation_seconds,
                   median_step_seconds=seconds, tokens_per_second=manifest.tokens_per_step/seconds,
                   predicted_step_seconds=report.predicted_makespan_ns/1e9,
                   peak_host_rss_preparation_bytes=prepared_rss,
                   peak_host_rss_execution_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                   pools_after_preparation=prepared_pools, pools_after_execution=pool_stats(stats_context),
                   measurements=asdict(measured), losses=losses, calibration=capabilities,
                   runtime_delta=_runtime_delta(runtime_before, adapter_statistics()),
                   calibration_attempts=calibration_attempts,
                   physical_statuses=[asdict(s) if hasattr(s, "__dataclass_fields__") else str(s) for s in physical_statuses])
    write_json(outdir / "result.json", summary)
    print("RESULT", json.dumps({key: summary[key] for key in
          ("status", "parameters", "trainable", "median_step_seconds", "tokens_per_second",
           "peak_host_rss_execution_bytes")}), flush=True)
    training.close()
    release_case_model(case, runtime=runtime)
    runtime.close()


if __name__ == "__main__":
    arguments = parse_args()
    arguments.outdir.mkdir(parents=True, exist_ok=True)
    try:
        run(arguments)
    except BaseException as error:
        write_json(arguments.outdir / "failure.json", dict(type=type(error).__name__,
                   message=str(error), traceback=traceback.format_exc()))
        raise
