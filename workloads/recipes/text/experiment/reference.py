"""Ordinary PyTorch reference loop, retained until legacy eager coverage migrates."""

from __future__ import annotations

import functools
import json
from contextlib import ExitStack
from pathlib import Path

import torch

from shadowspill.training import config, reset_parameters
from shadowspill.training._legacy_trainer import Trainer
from shadowspill.training.backends import PyTorch

from ..data.source import PackedUpdates, validation_update
from . import inputs, logging


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
        example, candidates, geometries, skipped = inputs.candidates(raw, data)
        trainer = stack.enter_context(
            Trainer(
                model,
                objective=objective,
                optimizer=config.resolve(raw["optimizer"]),
                optimizer_args=config.resolve(raw.get("optimizer_args", {})),
                schedules=config.resolve(raw.get("schedules", {})),
                microbatch_candidates=candidates,
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
        sizes = {name: p.numel() for name, p in trainer.model.named_parameters()}
        trainer.metric_reducer = logging.reducer(raw, sizes)
        logger = logging.open_logger(stack, root, raw)
        record_packing = logging.record_packing(stack, root, source, logger)

        eval_every = raw.get("eval_every", 0)
        evaluation = None
        if eval_every:
            count = raw.get("eval_microbatches", 1)
            if count < 1:
                raise ValueError(
                    "eval_microbatches must be positive when evaluation is enabled"
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
            eval_updates=1,
            checkpoint_every=raw.get("checkpoint_every", 0),
            checkpoint_dir=raw.get("checkpoint_dir"),
            checkpoint_weights=raw.get("checkpoint_weights", "master"),
            checkpoint_include_optimizer=raw.get("checkpoint_include_optimizer", True),
            keep_last=raw.get("keep_last", 3),
            startup_diagnostics=raw.get("startup_diagnostics", False),
        )
