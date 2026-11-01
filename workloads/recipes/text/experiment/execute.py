"""A fresh worker loads Plans, fills bounded tensors and runs the text loop.

With ``ranks`` workers, each is one rank of the Plan's parallelism: it trains
its share of every step's microbatches, writes its records and checkpoints
under ``rank-xxxxx/`` (its checkpoints under ``checkpoint_dir/rank-xxxxx/``
when the config names that directory), and logs through the distributed
logger, whose aggregate under ``aggregate/`` sums the ranks' shares into the
step's figures.
"""

import json
import math
import sys
import time
from contextlib import ExitStack
from datetime import timedelta
from pathlib import Path

from shadowspill import Plan, Trainer
from shadowspill.artifacts.tensors import registration_name, tensor_geometry
from shadowspill.training import config
from workloads.recipes.phases import settings

from . import group, inputs, logging, results


def checkpoint_directory(options, home, rank):
    """Where a rank saves its checkpoints: ``checkpoints/`` in its directory
    ``home``, or ``checkpoint_dir``, each of several ranks in its own
    ``rank-xxxxx/`` there, since every rank names its archive by the step."""
    if options.get("checkpoint_dir") is None:
        return home / "checkpoints"
    directory = Path(options["checkpoint_dir"])
    return directory / f"rank-{rank:05d}" if group.ranks(options) > 1 else directory


def resume_path(resume, directory):
    """A checkpoint path as given, or a step name (``step_00000100``) that
    every rank resolves to its own save in ``directory``, where it saves its
    checkpoints."""
    path = Path(resume)
    if len(path.parts) > 1 or path.suffix == ".ssp":
        return path
    return directory / f"{path.name}.ssp"


def run(directory):
    root, options = settings(directory)
    count = group.ranks(options)
    distributed = count > 1
    if distributed:
        import torch.distributed as dist

        dist.init_process_group("gloo", timeout=timedelta(seconds=7200))
    try:
        rank = dist.get_rank() if distributed else 0
        home = root / f"rank-{rank:05d}" if distributed else root
        home.mkdir(parents=True, exist_ok=True)
        checkpoints = checkpoint_directory(options, home, rank)
        choice = json.loads((root / "planning.json").read_text())
        plan = Plan.load(root / "plans/training.ssp")
        evaluation_path = root / "plans/evaluation.ssp"
        geometry = tensor_geometry(plan.program)
        sizes = {
            registration_name(item): math.prod(geometry[item["object_id"]]["shape"])
            for item in plan.program.bindings["registrations"]
            if item["parameter"]
        }
        # Model construction/settings belong to build. Only data, initialization
        # and ordinary Python logging/schedules are resolved inside this worker.
        data = config.resolve(options["data"])
        source = inputs.source(data, options, choice, rank)
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
                    rank_map={i: i for i in range(count)} if distributed else None,
                )
            )
            initialize = config.resolve(options.get("initialize"))
            if hasattr(initialize, "__enter__"):
                initialize = stack.enter_context(initialize)
            if options.get("resume"):
                trainer.load(
                    resume_path(options["resume"], checkpoints), fill_tensor=initialize
                )
            else:
                trainer.initialize(initialize)
            logger = logging.open_logger(
                stack, root, options, group=dist.group.WORLD if distributed else None
            )
            callbacks = [logging.record_packing(stack, home, source, logger)]
            if cadence := options.get("eval_every", 0):
                validation = inputs.validation(data, options, choice, rank)

                def evaluate(owner, result):
                    if result.step % cadence and result.step != options["steps"]:
                        return
                    started = time.perf_counter()
                    values = [owner.evaluate(args=[args]) for args in validation]
                    # This rank's share of the evaluation mean; the aggregate
                    # logger sums the ranks' shares.
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
                checkpoint_dir=checkpoints,
                keep_last=options.get("keep_last", 3),
                startup_diagnostics=options.get("startup_diagnostics", False),
            )
            results.save(last, home / "result.json")
            if distributed and rank == 0:
                results.save(last, root / "result.json")
    finally:
        if distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    run(sys.argv[1])
