"""Python entry point for the checkpoint recipe's ordinary saved phases."""

from argparse import Namespace
from collections.abc import Mapping
from os import fspath
from pathlib import Path

from workloads.recipes.phases import run_phases

from .arguments import parser
from .settings import configure


def run(config: Mapping[str, object], *, phase: str = "all") -> Path:
    """Run a commented Python configuration through the normal recipe workers.

    Keys use the CLI option names with underscores; paths can be Path objects.
    Omitted options use the same defaults as the CLI. A new run needs run_dir,
    checkpoint, frozen_state_id and data. Later phases can supply only run_dir
    and supported budget, step, resume or capture overrides; preparation choices
    are reloaded from config.json. Changed budgets require phase="plan".

    Each GPU phase runs in a fresh process. Planning remains CPU-only, and this
    entry point neither imports the model nor initializes CUDA in the caller.
    """
    if phase not in {"all", "build", "profile", "communicate", "plan", "execute"}:
        raise ValueError(f"unknown recipe phase: {phase!r}")
    if not isinstance(config, Mapping) or not config.get("run_dir"):
        raise ValueError("config needs a run_dir")
    if "phase" in config:
        raise ValueError("pass phase as run(config, phase=...), not a config key")

    # Reuse the CLI's defaults; no second set of recipe settings to keep in sync.
    defaults = vars(parser().parse_args([f"--run-dir={fspath(config['run_dir'])}"]))
    unknown = set(config) - defaults.keys()
    if unknown:
        raise ValueError(f"unknown recipe options: {', '.join(sorted(unknown))}")
    values = {**defaults, **config, "phase": phase}
    for name in ("run_dir", "checkpoint", "data", "resume"):
        if values[name] is not None:
            values[name] = fspath(values[name])
    root, options = configure(Namespace(**values))
    run_phases(
        "workloads.recipes.text.glm53_flash_lora", root, phase, ranks=options["ranks"]
    )
    return root
