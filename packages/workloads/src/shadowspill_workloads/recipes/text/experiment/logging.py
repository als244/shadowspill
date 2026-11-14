"""Text-specific CPU summaries and packing records, outside model tasks."""

import json

from shadowspill.training import config
from shadowspill.training.logging import Wandb
from shadowspill.training.observations import MetricSummary, parameter_scalars


def reducer(options, sizes):
    reduce_metrics = config.resolve(options.get("metric_reducer"))
    omitted = tuple(
        options.get("omit_parameter_metrics", ["grad_squared_share/", "param_norm/"])
    )

    def reduce_observations(observations):
        values = tuple(value for value in observations.metrics if value is not None)
        summary = (
            reduce_metrics(values)
            if reduce_metrics is not None and values
            else MetricSummary()
        )
        scalars = dict(summary.scalars)
        scalars.update(
            (name, value)
            for name, value in parameter_scalars(
                observations.parameter_metrics, sizes
            ).items()
            if not name.startswith(omitted)
        )
        return MetricSummary(scalars, summary.tables)

    return reduce_observations


def sum_shares(records):
    """The ranks' records as the step's: what the default reducer adds, and
    the evaluation loss, which every rank of these recipes reports as its share
    of one mean over all ranks' held-out tokens."""
    from shadowspill.training.logging import sum_rank_records

    combined = dict(sum_rank_records(records))
    if all("eval/loss" in record for record in records):
        combined["eval/loss"] = sum(record["eval/loss"] for record in records)
    return combined


def open_logger(stack, root, options, group=None):
    """The run's logger: W&B when ``wandb_project`` says so; with a rank
    ``group``, the distributed logger, which keeps a W&B run per rank and one
    aggregate beside ``aggregate/metrics.jsonl`` under ``root``."""
    arguments = dict(options.get("wandb", {}))
    arguments.setdefault("name", root.name)
    arguments.setdefault("mode", options.get("wandb_mode", "online"))
    if group is not None:
        from shadowspill.training.logging import DistributedLogger

        wandb = (
            {"project": options["wandb_project"], **arguments}
            if options.get("wandb_project")
            else None
        )
        return stack.enter_context(
            DistributedLogger(group, run_dir=root, wandb=wandb, reduce=sum_shares)
        )
    if not options.get("wandb_project"):
        return None
    return stack.enter_context(
        Wandb(project=options["wandb_project"], run_dir=root, **arguments)
    )


def record_packing(stack, root, source, logger):
    stream = stack.enter_context((root / "packing.jsonl").open("a", buffering=1))

    def record(engine, result):
        stream.write(
            json.dumps(
                {
                    "step": result.step,
                    "microbatches": source.last_documents,
                    **source.last_stats,
                }
            )
            + "\n"
        )
        emit(
            root,
            logger,
            {
                "step": result.step,
                "train/tokens_per_second": source.last_stats["trained_tokens"]
                / result.seconds,
                "packing/trained_tokens_total": source.trained_tokens,
                **{"packing/" + k: v for k, v in source.last_stats.items()},
            },
        )

    return record


def emit(root, logger, record):
    with (root / "metrics.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    if logger is not None:
        logger(record)
