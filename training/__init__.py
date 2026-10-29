"""Repository text-recipe commands and configs; generic APIs: shadowspill.training.

    from training import run

    result = run("finetune.json", run_dir="runs/qwen3", steps=1000)

``run`` takes a config, a path or a JSON-like mapping, and the overrides the
command line takes as keyword arguments (a key with dots as
``**{"schedules.optimizer_group.0.lr.total_steps": 300}``), runs every phase
or the one ``phase`` names, and returns the final step's result after
execution.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def run(
    config: str | Path | Mapping[str, Any], *, phase: str = "all", **overrides: Any
) -> Any:
    from workloads.recipes.text.run import run_config

    return run_config(
        config,
        [f"{key}={json.dumps(value)}" for key, value in overrides.items()],
        phase=phase,
    )


__all__ = ["run"]
