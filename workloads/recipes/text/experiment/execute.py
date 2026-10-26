"""A fresh worker loads Plans, fills bounded tensors and runs the text loop."""

import json
import math
import sys
import time
from contextlib import ExitStack

from shadowspill import Plan, Trainer
from shadowspill.artifacts.tensors import registration_name, tensor_geometry
from shadowspill.training import config
from workloads.recipes.phases import settings

from . import inputs, logging, results


def run(directory):
    root, options = settings(directory)
    choice = json.loads((root / "planning.json").read_text())
    plan = Plan.load(root / "plans/training.ssp")
    evaluation_path = root / "plans/evaluation.ssp"
    geometry = tensor_geometry(plan.program)
    sizes = {
        registration_name(item): math.prod(geometry[item["object_id"]]["shape"])
        for item in plan.program.bindings["registrations"]
        if item["parameter"]
    }
    # Model construction/settings belong to build. Only data, initialization and
    # ordinary Python logging/schedules are resolved inside this fresh worker.
    data = config.resolve(options["data"])
    source = inputs.source(data, options, choice)
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
            )
        )
        initialize = config.resolve(options.get("initialize"))
        if hasattr(initialize, "__enter__"):
            initialize = stack.enter_context(initialize)
        if options.get("resume"):
            trainer.load(options["resume"], fill_tensor=initialize)
        else:
            trainer.initialize(initialize)
        logger = logging.open_logger(stack, root, options)
        callbacks = [logging.record_packing(stack, root, source, logger)]
        if cadence := options.get("eval_every", 0):
            validation = inputs.validation(data, options, choice)

            def evaluate(owner, result):
                if result.step % cadence and result.step != options["steps"]:
                    return
                started = time.perf_counter()
                values = [owner.evaluate(args=[args]) for args in validation]
                logging.emit(
                    root,
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
            run_dir=root,
            logger=logger,
            callbacks=callbacks,
            log_every=options.get("log_every", 1),
            tables_every=options.get("metric_tables_every", 100),
            checkpoint_every=options.get("checkpoint_every", 0),
            checkpoint_dir=options.get("checkpoint_dir"),
            keep_last=options.get("keep_last", 3),
        )
        results.save(last, root / "result.json")


if __name__ == "__main__":
    run(sys.argv[1])
