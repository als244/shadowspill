"""Configured text runs: independent phases, or an ordinary PyTorch reference."""

import argparse
import copy
import difflib
import json
from collections.abc import Mapping
from pathlib import Path

from shadowspill.training import config
from workloads.recipes.phases import run_phases

#: Every key a config may hold; a key outside it is refused by name.
KNOWN_KEYS = frozenset(
    {
        "run_dir",
        "engine",
        "backend",
        "settings",
        "model",
        "checkpoint",
        "data",
        "objective",
        "objective_args",
        "optimizer",
        "optimizer_args",
        "master_dtype",
        "grad_dtype",
        "build",
        "initialize",
        "resume",
        "steps",
        "schedules",
        "max_seq_len",
        "max_tokens_per_step",
        "max_tokens_per_microbatch",
        "planning_min_tokens_per_microbatch",
        "planning_max_tokens_per_microbatch",
        "seed",
        "eval_every",
        "eval_microbatches",
        "checkpoint_every",
        "checkpoint_dir",
        "keep_last",
        "log_every",
        "metric_tables_every",
        "parameter_norms_every",
        "parameter_metrics",
        "metric_reducer",
        "omit_parameter_metrics",
        "frozen_state_id",
        "startup_diagnostics",
        "wandb_project",
        "wandb_mode",
        "wandb",
        "warmup_seconds",
        "measure_seconds",
        "external_headroom_mib",
        "execution_gib",
        "spill_gib",
        "plots",
        "checkpoint_weights",
        "checkpoint_include_optimizer",
        "device",
        "compile",
        "ranks",
        "shares",
        "hosts",
        "port",
    }
)

#: What a config must say; everything else has a default or is optional.
REQUIRED_KEYS = ("run_dir", "data", "max_seq_len", "max_tokens_per_step", "steps")

#: What a config may leave out.
DEFAULTS = {
    "objective": "@workloads.recipes.text.objectives:next_token_prediction",
    "optimizer": "@mlops.optim:AdamW",
}

#: Of a checkpoint entry, what only the model takes and what only the initializer takes.
MODEL_ONLY = frozenset({"implementation", "dtype", "lora"})
INITIALIZER_ONLY = frozenset({"seed"})


def check_keys(raw):
    """Refuse a key the runner does not know, naming the nearest it does."""
    unknown = sorted(set(raw) - KNOWN_KEYS)
    if unknown:
        hints = []
        for key in unknown:
            matches = difflib.get_close_matches(key, KNOWN_KEYS, n=1)
            hints.append(f"{key} (did you mean {matches[0]}?)" if matches else key)
        raise ValueError("unknown config keys: " + ", ".join(hints))


def expand_checkpoint(raw):
    """A ``checkpoint`` entry -- a family and a local Hugging Face checkpoint
    directory, with ``implementation``, ``dtype`` and ``lora`` for the model,
    ``seed`` for the initializer (the run's seed otherwise) and any other
    option, such as ``max_seq_len`` (the run's otherwise), for both -- stands
    for ``model`` built at the checkpoint's configuration and ``initialize``
    streaming it in. The two explicit entries remain for anyone who wants them."""
    entry = raw.get("checkpoint")
    if entry is None:
        return raw
    if "model" in raw or "initialize" in raw:
        raise ValueError(
            "checkpoint stands for model and initialize; give one form or the other"
        )
    if not {"family", "directory"} <= set(entry):
        raise ValueError("a checkpoint entry names a family and directory")
    shared = {k: v for k, v in entry.items() if k not in MODEL_ONLY | INITIALIZER_ONLY}
    if "max_seq_len" not in shared and "max_seq_len" in raw:
        shared["max_seq_len"] = raw["max_seq_len"]
    expanded = dict(raw)
    expanded["model"] = {
        "@call": "workloads.models.catalog:build_from_checkpoint",
        **shared,
        **{k: entry[k] for k in ("implementation", "dtype", "lora") if k in entry},
    }
    expanded["initialize"] = {
        "@call": "workloads.recipes.text.initialization:from_checkpoint",
        **shared,
        "seed": entry.get("seed", raw.get("seed", 0)),
    }
    return expanded


def expand_defaults(raw):
    """``data`` given as a directory alone is the packed documents it holds;
    the objective defaults to next-token prediction and the optimizer to the
    mlops AdamW (its ``lr`` still said in ``optimizer_args``)."""
    expanded = dict(raw)
    if isinstance(expanded.get("data"), str):
        expanded["data"] = {
            "@call": "workloads.recipes.text.data.packed:PackedTokens",
            "directory": expanded["data"],
        }
    for key, value in DEFAULTS.items():
        expanded.setdefault(key, value)
    return expanded


def check_required(raw):
    missing = [key for key in REQUIRED_KEYS if key not in raw]
    if missing:
        raise ValueError(f"a config needs {', '.join(missing)}")
    if not any(key in raw for key in ("checkpoint", "model")):
        raise ValueError("a config names a checkpoint or a model")


def load_config(source, overrides=()):
    """The config at a path, or a JSON-like mapping, with overrides applied,
    its keys checked, its checkpoint entry and defaults expanded."""
    if isinstance(source, Mapping):
        raw = config.override(copy.deepcopy(dict(source)), list(overrides))
    else:
        raw = config.load(source, list(overrides))
    check_keys(raw)
    raw = expand_defaults(expand_checkpoint(raw))
    check_required(raw)
    return raw


def run_config(path, overrides=(), *, phase="all"):
    raw = load_config(path, overrides)
    if raw.get("backend"):
        raise ValueError("use engine='pytorch' or engine='shadowspill', not backend")
    ranks = raw.get("ranks", 1)
    if type(ranks) is not int or ranks < 1:
        raise ValueError("ranks must be a positive integer")
    if raw.get("engine") == "pytorch":
        if phase != "all":
            raise ValueError("ordinary PyTorch reference execution has no saved phases")
        if ranks > 1:
            raise ValueError(
                "the PyTorch reference runs one worker; "
                "ranks needs the shadowspill engine"
            )
        from .reference.run import run_config as run_reference

        return run_reference(path, overrides, options=raw)
    if raw.get("engine", "shadowspill") != "shadowspill":
        raise ValueError("use engine='shadowspill' with explicit phase settings")
    if phase not in {"all", "build", "profile", "communicate", "plan", "execute"}:
        raise ValueError(f"unknown phase: {phase}")
    if phase == "communicate" and ranks == 1:
        raise ValueError("one worker measures its host copies in the profile phase")
    if phase in {"all", "execute"} and not (raw.get("initialize") or raw.get("resume")):
        raise ValueError("supply initialize=(chunk, destination) or resume=checkpoint")
    if raw.get("parameter_metrics") is not None:
        raise ValueError(
            "use parameter_norms_every; raw metric callbacks are not tasks"
        )
    if raw.get("checkpoint_weights", "master") != "master" or not raw.get(
        "checkpoint_include_optimizer", True
    ):
        raise ValueError("saved-Plan resume checkpoints retain full training state")
    root = Path(raw["run_dir"]).resolve()
    if phase in {"all", "build"} and (root / "candidates.json").exists():
        raise ValueError("build exists; choose a later phase or a fresh run directory")
    root.mkdir(parents=True, exist_ok=True)
    raw["run_dir"] = str(root)
    (root / "config.json").write_text(json.dumps(raw, indent=2) + "\n")
    from .experiment.hosts import check_hosts, run_hosts

    hosts = check_hosts(raw)
    if hosts is None:
        run_phases("workloads.recipes.text.experiment", root, phase, ranks=ranks)
    else:
        # The phases on every host read the group's size from the config.
        raw["ranks"] = sum(host.get("ranks", 1) for host in hosts)
        (root / "config.json").write_text(json.dumps(raw, indent=2) + "\n")
        run_hosts(root, raw, phase, hosts)
    if phase in {"all", "execute"}:
        from .experiment.results import load

        for record in (root / "aggregate/wandb_run.json", root / "wandb_run.json"):
            if record.exists() and (url := json.loads(record.read_text()).get("url")):
                print(f"W&B run: {url}", flush=True)
                break
        return load(root / "result.json")
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument(
        "--phase",
        choices=["all", "build", "profile", "communicate", "plan", "execute"],
        default="all",
    )
    args = parser.parse_args(argv)
    run_config(args.config, args.overrides, phase=args.phase)


if __name__ == "__main__":
    main()
