"""The supplied text experiment recipe, composed from generic training APIs."""

from __future__ import annotations

import functools
import json
from contextlib import ExitStack
from pathlib import Path

import torch

from shadowspill.training import Trainer, config, reset_parameters
from shadowspill.training.backends import PyTorch
from shadowspill.training.logging import Wandb
from shadowspill.training.observations import MetricSummary, parameter_scalars

from .source import PackedUpdates, candidate_data, validation_update


def run_config(path, overrides=()):
    """Resolve this recipe's JSON inputs, then call the generic Trainer."""
    raw = config.load(path, list(overrides))
    root = Path(raw["run_dir"])
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(raw, indent=2) + "\n")
    with ExitStack() as stack:
        backend = config.resolve(raw["backend"]) if "backend" in raw else PyTorch()
        # The runtime context precedes settings, model resources and device state.
        stack.enter_context(backend)
        config.resolve(raw.get("settings", []))
        torch.manual_seed(raw.get("seed", 0))
        model = config.resolve(raw["model"])
        data = config.resolve(raw["data"])
        objective = functools.partial(
            config.resolve(raw["objective"]),
            **config.resolve(raw.get("objective_args", {})),
        )
        bounds = dict(
            min_tokens_per_microbatch=raw.get("planning_min_tokens_per_microbatch"),
            max_tokens_per_microbatch=raw.get("planning_max_tokens_per_microbatch"),
        )
        fixed = raw.get("max_tokens_per_microbatch")
        if fixed is not None:
            if any(value is not None for value in bounds.values()):
                raise ValueError("choose a fixed text microbatch size or search bounds")
            bounds = dict(
                min_tokens_per_microbatch=fixed, max_tokens_per_microbatch=fixed
            )
        example, candidates, geometries, skipped = candidate_data(
            data,
            max_seq_len=raw["max_seq_len"],
            max_tokens_per_step=raw["max_tokens_per_step"],
            **bounds,
        )
        trainer = stack.enter_context(
            Trainer(
                model,
                objective=objective,
                optimizer=config.resolve(raw["optimizer"]),
                optimizer_args=config.resolve(raw.get("optimizer_args", {})),
                schedules=config.resolve(raw.get("schedules", {})),
                microbatches=candidates,
                backend=backend,
                master_dtype=config.resolve(raw.get("master_dtype")),
                grad_dtype=config.resolve(raw.get("grad_dtype")),
                parameter_metrics=config.resolve(raw.get("parameter_metrics")),
            )
        )
        seed = raw.get("seed", 0)

        def initialize(module):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                reset_parameters(module)

        meta = any(value.is_meta for value in (*model.parameters(), *model.buffers()))
        trainer.prepare(
            example,
            initialize=initialize if meta else None,
            checkpoint=raw.get("resume"),
        )
        name = trainer.selected_candidate
        tokens, accumulation = geometries[name]
        (root / "planning.json").write_text(
            json.dumps(
                {
                    "candidate": name,
                    "tokens_per_microbatch": tokens,
                    "microbatches": accumulation,
                    "skipped": skipped,
                },
                indent=2,
            )
            + "\n"
        )
        source = PackedUpdates(
            data,
            name=name,
            tokens=tokens,
            accumulation=accumulation,
            max_seq_len=raw["max_seq_len"],
        )
        reducer = config.resolve(raw.get("metric_reducer"))
        sizes = {name: p.numel() for name, p in trainer.model.named_parameters()}
        omitted = tuple(
            raw.get(
                "omit_parameter_metrics",
                [
                    "grad_squared_share/",
                    "param_norm/",
                ],
            )
        )

        def reduce_observations(observations):
            values = tuple(value for value in observations.metrics if value is not None)
            summary = (
                reducer(values) if reducer is not None and values else MetricSummary()
            )
            scalars = dict(summary.scalars)
            if observations.parameter_metrics:
                scalars.update(
                    {
                        name: value
                        for name, value in parameter_scalars(
                            observations.parameter_metrics, sizes
                        ).items()
                        if not name.startswith(omitted)
                    }
                )
            return MetricSummary(scalars, summary.tables)

        trainer.metric_reducer = reduce_observations
        logger = None
        if raw.get("wandb_project"):
            options = dict(raw.get("wandb", {}))
            options.setdefault("name", root.name)
            options.setdefault("mode", raw.get("wandb_mode", "online"))
            logger = stack.enter_context(
                Wandb(
                    project=raw["wandb_project"],
                    run_dir=root,
                    **options,
                )
            )
        packing_log = stack.enter_context(
            (root / "packing.jsonl").open("a", buffering=1)
        )

        def record_packing(engine, result):
            packing_log.write(
                json.dumps(
                    {
                        "step": result.step,
                        "microbatches": source.last_documents,
                        **source.last_stats,
                    }
                )
                + "\n"
            )
            record = {
                "step": result.step,
                "train/tokens_per_second": source.last_stats["trained_tokens"]
                / result.seconds,
                "packing/trained_tokens_total": source.trained_tokens,
                **{"packing/" + k: v for k, v in source.last_stats.items()},
            }
            with (root / "metrics.jsonl").open("a") as output:
                output.write(json.dumps(record) + "\n")
            if logger is not None:
                logger(record)

        eval_every = raw.get("eval_every", 0)
        evaluation = None
        if eval_every:
            count = raw.get("eval_batches", 1)
            if count < 1:
                raise ValueError(
                    "eval_batches must be positive when evaluation is enabled"
                )
            evaluation = [
                validation_update(
                    data,
                    name=name,
                    tokens=tokens,
                    max_seq_len=raw["max_seq_len"],
                    microbatches=count,
                )
            ]
        return trainer.fit(
            source,
            steps=raw["steps"],
            run_dir=root,
            logger=logger,
            callbacks=(record_packing,),
            log_every=raw.get("log_every", 1),
            tables_every=raw.get("metric_tables_every", 100),
            eval_data=evaluation,
            eval_every=eval_every,
            eval_batches=1,
            checkpoint_every=raw.get("checkpoint_every", 0),
            checkpoint_dir=raw.get("checkpoint_dir"),
            checkpoint_weights=raw.get("checkpoint_weights", "master"),
            keep_last=raw.get("keep_last", 3),
            startup_diagnostics=raw.get("startup_diagnostics", False),
        )
