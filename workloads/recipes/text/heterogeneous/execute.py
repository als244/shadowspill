"""One worker per rank, each with its node's Plan and its share of every step."""

import json
import math
import sys
import time
from contextlib import ExitStack
from datetime import timedelta

import torch.distributed as dist

from shadowspill import Plan, Trainer
from shadowspill.artifacts.tensors import registration_name, tensor_geometry
from shadowspill.training import config
from workloads.recipes.phases import settings
from workloads.recipes.text.experiment import logging, results
from workloads.recipes.text.experiment.execute import resume_path

from . import data as units
from .run import check_section


def run(directory):
    root, options = settings(directory)
    _, shares, _ = check_section(options)
    dist.init_process_group("gloo", timeout=timedelta(seconds=7200))
    try:
        rank = dist.get_rank()
        home = root / f"rank-{rank:05d}"
        home.mkdir(parents=True, exist_ok=True)
        choice = json.loads((root / "planning.json").read_text())
        plan = Plan.load(root / "plans/training.ssp")
        evaluation_path = root / "plans/evaluation.ssp"
        geometry = tensor_geometry(plan.program)
        sizes = {
            registration_name(item): math.prod(geometry[item["object_id"]]["shape"])
            for item in plan.program.bindings["registrations"]
            if item["parameter"]
        }
        corpus = config.resolve(options["data"])
        step_units = options["max_tokens_per_step"] // options["max_seq_len"]
        shared = dict(
            name=choice["candidate"],
            max_seq_len=options["max_seq_len"],
            capacity=choice["tokens_per_microbatch"],
            step_units=step_units,
            shares=shares,
            rank=rank,
        )
        source = units.UnitUpdates(corpus, **shared)
        with ExitStack() as stack:
            trainer = stack.enter_context(
                Trainer(
                    plan,
                    evaluation_plan=Plan.load(evaluation_path)
                    if evaluation_path.exists()
                    else None,
                    schedules=config.resolve(options.get("schedules", {})),
                    metric_reducer=logging.reducer(options, sizes),
                    parameter_norms_every=options.get("parameter_norms_every", 0),
                    frozen_state_id=options.get("frozen_state_id"),
                    rank_map={i: i for i in range(options["ranks"])},
                )
            )
            initialize = config.resolve(options.get("initialize"))
            if hasattr(initialize, "__enter__"):
                initialize = stack.enter_context(initialize)
            if options.get("resume"):
                trainer.load(
                    resume_path(options["resume"], home), fill_tensor=initialize
                )
            else:
                trainer.initialize(initialize)
            logger = logging.open_logger(stack, root, options, group=dist.group.WORLD)
            callbacks = [logging.record_packing(stack, home, source, logger)]
            if cadence := options.get("eval_every", 0):
                held_out = units.validation(
                    corpus, steps=options.get("eval_microbatches", 1), **shared
                )

                def evaluate(owner, result):
                    if result.step % cadence and result.step != options["steps"]:
                        return
                    started = time.perf_counter()
                    values = [owner.evaluate(args=[args]) for args in held_out]
                    logging.emit(
                        home,
                        logger,
                        {
                            "step": result.step,
                            "eval/loss": sum(float(value[0]) for value in values),
                            "eval/seconds": time.perf_counter() - started,
                        },
                    )

                callbacks.append(evaluate)
            last = trainer.fit(
                source,
                steps=options["steps"],
                run_dir=home,
                logger=logger,
                callbacks=callbacks,
                log_every=options.get("log_every", 1),
                tables_every=options.get("metric_tables_every", 100),
                checkpoint_every=options.get("checkpoint_every", 0),
                checkpoint_dir=options.get("checkpoint_dir"),
                keep_last=options.get("keep_last", 3),
                startup_diagnostics=options.get("startup_diagnostics", False),
            )
            results.save(last, home / "result.json")
            if rank == 0:
                results.save(last, root / "result.json")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    run(sys.argv[1])
