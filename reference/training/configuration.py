"""Validate model, objective and data settings for the independent text reference."""

import copy
import difflib
from collections.abc import Mapping

from shadowspill.training import config

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
        "eval_tokens",
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
        "nodes",
        "port",
        "heterogeneous",
    }
)

#: What a config must say; everything else has a default or is optional.
REQUIRED_KEYS = ("run_dir", "data", "max_seq_len", "max_tokens_per_step", "steps")

#: What a config may leave out.
DEFAULTS = {
    "objective": "@shadowspill_workloads.objectives.next_token:next_token_prediction",
    "optimizer": "@mlops.optim:AdamW",
}

#: Of a checkpoint entry, what only the model takes and what only the initializer takes.
MODEL_ONLY = frozenset({"implementation", "dtype", "lora"})
INITIALIZER_ONLY = frozenset({"seed"})


#: Keys a config may still carry under an earlier name, with what replaced them.
RENAMED = {
    "eval_microbatches": "eval_tokens, the held-out tokens each evaluation covers",
}


def check_keys(raw):
    """Refuse a key the runner does not know, naming the nearest it does."""
    unknown = sorted(set(raw) - KNOWN_KEYS)
    if unknown:
        hints = []
        for key in unknown:
            matches = difflib.get_close_matches(key, KNOWN_KEYS, n=1)
            if key in RENAMED:
                hints.append(f"{key} (now {RENAMED[key]})")
            else:
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
        "@call": "shadowspill_workloads.models.catalog:build_from_checkpoint",
        **shared,
        **{k: entry[k] for k in ("implementation", "dtype", "lora") if k in entry},
    }
    expanded["initialize"] = {
        "@call": "shadowspill_workloads.initialization.decoder:from_checkpoint",
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
            "@call": "shadowspill_workloads.inputs.text.packed:PackedTokens",
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
