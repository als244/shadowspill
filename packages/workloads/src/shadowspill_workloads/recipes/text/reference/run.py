"""Configured ordinary PyTorch training; no preparation backend or runtime pool."""

import json
from contextlib import ExitStack
from pathlib import Path

import torch

from shadowspill.training import config

from ..experiment import inputs, logging
from .engine import EagerStep
from .initialization import initializer
from .loop import fit


def run_config(path, overrides=(), *, options=None):
    options = config.load(path, list(overrides)) if options is None else dict(options)
    if options.get("startup_diagnostics"):
        raise ValueError("the PyTorch reference has no startup step timelines")
    root = Path(options["run_dir"])
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(options, indent=2) + "\n")
    config.resolve(options.get("settings", []))
    seed = options.get("seed", 0)
    torch.manual_seed(seed)
    model = config.resolve(options["model"])
    data = config.resolve(options["data"])
    _, _, geometries, skipped = inputs.candidates(options, data)
    if len(geometries) != 1:
        raise ValueError("PyTorch needs one microbatch candidate; select a fixed size")
    name, (tokens, count) = next(iter(geometries.items()))
    choice = {
        "candidate": name,
        "tokens_per_microbatch": tokens,
        "microbatches": count,
        "skipped": skipped,
    }
    (root / "planning.json").write_text(json.dumps(choice, indent=2) + "\n")

    meta = any(value.is_meta for value in (*model.parameters(), *model.buffers()))
    with initializer(options) as initialize:
        engine = EagerStep(
            model, options, initialize if meta or options.get("initialize") else None
        )
    sizes = {name: p.numel() for name, p in engine.model.named_parameters()}
    source = inputs.source(data, options, choice)
    with ExitStack() as stack:
        logger = logging.open_logger(stack, root, options)
        return fit(
            engine,
            source,
            data,
            choice,
            root,
            options,
            logging.reducer(options, sizes),
            logger,
            logging.record_packing(stack, root, source, logger),
        )
