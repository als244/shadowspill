"""Persist and validate choices without importing tensor frameworks."""

import importlib
import json
import math
from collections.abc import Mapping
from os import PathLike, fspath
from pathlib import Path, PurePosixPath

PHASES = {"all", "build", "profile", "communicate", "plan", "execute"}
DEFAULTS = dict(
    run_dir=None,
    steps=5,
    ranks=1,
    preparation_ranks=1,
    execution_gib=28,
    spill_gib=144,
    external_headroom_mib=512,
    warmup_seconds=0.05,
    measure_seconds=0.18,
    refresh_profiles=False,
    plots=True,
    eval_every=2,
    eval_updates=1,
    checkpoint_every=2,
    keep_last=2,
    program_workers=8,
    search_workers=8,
    resolution_options=None,
    tail_save_counts=None,
    resume=None,
    wandb_project=None,
    pool_memory=None,
    prepared_from=None,
    nsys_steps=0,
    nsys_start_step=2,
)
MUTABLE = {
    "steps",
    "resume",
    "plots",
    "search_workers",
    "program_workers",
    "resolution_options",
    "tail_save_counts",
    "execution_gib",
    "spill_gib",
    "pool_memory",
    "refresh_profiles",
    "checkpoint_every",
    "keep_last",
    "nsys_steps",
    "nsys_start_step",
}


def adapter(name):
    if not isinstance(name, str) or not name:
        raise ValueError("recipe must name an importable adapter module")
    return importlib.import_module(name)


def configure(recipe, config, phase):
    if phase not in PHASES:
        raise ValueError(f"unknown recipe phase: {phase!r}")
    if not isinstance(config, Mapping) or not config.get("run_dir"):
        raise ValueError("config needs a run_dir")
    if "phase" in config:
        raise ValueError("pass phase as run(..., phase=...), not a config key")
    definition = adapter(recipe)
    defaults = {**DEFAULTS, **definition.defaults}
    unknown = config.keys() - defaults.keys()
    if unknown:
        raise ValueError(f"unknown recipe options: {', '.join(sorted(unknown))}")
    requested = {
        key: fspath(value) if isinstance(value, PathLike) else value
        for key, value in config.items()
    }
    for key in (
        "pool_memory",
        "resume",
        "prepared_from",
        *getattr(definition, "path_options", ()),
    ):
        if requested.get(key):
            requested[key] = str(Path(requested[key]).expanduser().resolve())
    root = Path(requested["run_dir"]).expanduser().resolve()
    requested["run_dir"] = str(root)
    path = root / "config.json"
    reused = not path.exists() and requested.get("prepared_from")
    if reused and phase != "plan":
        raise ValueError("start a prepared_from run with phase='plan'")
    if path.exists() or reused:
        if phase == "all":
            raise ValueError("use a fresh run directory for a new build")
        if reused:
            from .reuse import source_settings

            options = source_settings(requested["prepared_from"], root)
        else:
            options = json.loads(path.read_text())
        options = {**defaults, **options}
        if options.get("prepared_from") and phase not in {"plan", "execute"}:
            raise ValueError("a prepared_from run may only plan or execute")
        if options["recipe"] != recipe:
            raise ValueError("saved run belongs to a different recipe")
        for key, value in requested.items():
            if value == options.get(key):
                continue
            if key not in MUTABLE | set(getattr(definition, "runtime_options", ())):
                raise ValueError(
                    f"changed {key} requires a fresh run directory and build"
                )
            if key == "refresh_profiles" and phase != "profile":
                raise ValueError("changed refresh_profiles requires phase='profile'")
            if (
                key
                in {
                    "execution_gib",
                    "spill_gib",
                    "resolution_options",
                    "tail_save_counts",
                    "pool_memory",
                }
                and phase != "plan"
            ):
                raise ValueError(
                    "changed budgets or planning choices require phase='plan'"
                )
            options[key] = value
    else:
        if phase not in {"all", "build"}:
            raise ValueError("run the build phase first")
        options = {**defaults, **requested, "recipe": recipe}
    for key in (
        "steps",
        "ranks",
        "preparation_ranks",
        "eval_updates",
        "keep_last",
        "program_workers",
        "nsys_start_step",
    ):
        if type(options[key]) is not int or options[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("eval_every", "checkpoint_every", "search_workers", "nsys_steps"):
        if type(options[key]) is not int or options[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if options["nsys_steps"]:
        first = options["nsys_start_step"]
        last = first + options["nsys_steps"] - 1
        if last > options["steps"]:
            raise ValueError("steps must cover the requested Nsight capture")
        for cadence in (options["eval_every"], options["checkpoint_every"]):
            if cadence and any(step % cadence == 0 for step in range(first, last)):
                raise ValueError(
                    "evaluation/checkpoint cadence must lie outside capture"
                )
    for key in ("execution_gib", "spill_gib", "measure_seconds"):
        if (
            isinstance(options[key], bool)
            or not isinstance(options[key], int | float)
            or not math.isfinite(options[key])
            or options[key] <= 0
        ):
            raise ValueError(f"{key} must be positive")
    if options["warmup_seconds"] < 0 or options["external_headroom_mib"] < 0:
        raise ValueError("warmup and headroom must be nonnegative")
    if type(options.get("refresh_profiles", False)) is not bool:
        raise ValueError("refresh_profiles must be a boolean")
    if options["preparation_ranks"] not in (1, options["ranks"]):
        raise ValueError("preparation requires one worker or the entire worker group")
    if options.get("tail_save_counts") is not None:
        from shadowspill.planner import validate_tail_save_counts

        options["tail_save_counts"] = list(
            validate_tail_save_counts(options["tail_save_counts"])
        )
    definition.validate(options)
    if not path.exists() and not reused:
        candidates = list(definition.candidates(options))
        paths = set()
        for candidate in candidates:
            name = candidate["path"]
            parsed = PurePosixPath(name)
            if (
                not name
                or parsed.is_absolute()
                or ".." in parsed.parts
                or name in paths
            ):
                raise ValueError("candidate paths must be unique relative paths")
            paths.add(name)
            if (
                type(candidate["num_examples"]) is not int
                or candidate["num_examples"] < 1
            ):
                raise ValueError("candidate num_examples must be positive")
        if not candidates:
            raise ValueError("at least one build candidate is required")
        options["candidates"] = candidates
    if phase == "communicate" and options["ranks"] == 1:
        raise ValueError("single-GPU profiling already measures host communication")
    root.mkdir(parents=True, exist_ok=True)
    if reused:
        from .reuse import link_artifacts

        link_artifacts(options["prepared_from"], root)
    path.write_text(json.dumps(options, indent=2) + "\n")
    return root, options
